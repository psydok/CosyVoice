# CosyVoice3 Architecture

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CosyVoice3 Inference Pipeline                     │
│                                                                          │
│  "Hello world" ──► [Frontend] ──► [LLM] ──► [Flow/DiT] ──► [Vocoder] ──► 🔊
│                     text norm     tokens    mel-spec      waveform       │
│                     tokenize      25 Hz     50 Hz         24 kHz         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Three-Stage Architecture

```
Stage 1: LLM (Autoregressive)          Stage 2: Flow/DiT (Diffusion)         Stage 3: Vocoder (Conv)
┌──────────────────────────┐      ┌────────────────────────────────┐    ┌─────────────────────────┐
│     CosyVoice3LM         │      │   CausalMaskedDiffWithDiT      │    │  CausalHiFTGenerator    │
│     (Qwen2-based)        │      │                                │    │                         │
│                          │      │  Speech Tokens (6561 vocab)    │    │  Mel (80-dim, 50Hz)     │
│  Text ──► Qwen2 ──► Token│─────►│  ──► Embed ──► PreLookahead    │    │  ──► Conv1d(80,512)     │
│           896-dim   6561 │      │  ──► Repeat ×2 (mel ratio)     │    │  ──► Upsample ×8        │
│                          │      │  ──► DiT Decoder (10 steps)    │────►│  ──► Upsample ×5        │
│  25 tokens/sec           │      │  ──► Mel Spectrogram           │    │  ──► Upsample ×3        │
│  ~40ms per token         │      │                                │    │  ──► iSTFT(n_fft=16)    │
│                          │      │  50 mel frames/sec             │    │  ──► Waveform            │
└──────────────────────────┘      └────────────────────────────────┘    │                         │
                                                                        │  24000 samples/sec      │
                                                                        └─────────────────────────┘
```

## Detailed Component Breakdown

### 1. Frontend: CosyVoiceFrontEnd

```
Input Text
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Text Normalization                                   │
│  ├── wetext (zh/en/ru/etc.)                          │
│  └── Sentence splitting                               │
│                                                       │
│  Text Tokenization (Qwen2 tiktoken)                  │
│  ├── Multilingual: zh, en, ja, ko, de, es, fr, it, ru│
│  └── Special: <|endofprompt|> marker (required)       │
│                                                       │
│  Speaker Embedding Extraction                         │
│  ├── Input: 16kHz waveform                           │
│  ├── Features: 80-bin fbank (Kaldi)                  │
│  ├── Model: CampPlus ONNX                            │
│  └── Output: 192-dim normalized embedding             │
│                                                       │
│  Speech Token Extraction (for prompt audio)           │
│  ├── Input: 16kHz waveform                           │
│  ├── Features: 128-bin Whisper mel                   │
│  ├── Model: speech_tokenizer_v3.onnx (FSQ v3)       │
│  └── Output: tokens from 6561-vocab @ 25Hz           │
└──────────────────────────────────────────────────────┘
```

### 2. LLM: CosyVoice3LM

```
┌─────────────────────────────────────────────────────────────────┐
│  CosyVoice3LM (inherits Qwen2LM ← TransformerLM)              │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │ Text Embedding│   │Speech Embedding│  │ LLM Task Embedding│  │
│  │ Qwen2 vocab   │   │ 6761 → 896    │  │ [SOS, TaskID]     │  │
│  │ → 896-dim     │   │               │  │ → 896-dim         │  │
│  └──────┬───────┘   └──────┬────────┘  └────────┬──────────┘  │
│         │                   │                     │             │
│         └───────────┬───────┘─────────────────────┘             │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────┐        │
│  │           Qwen2ForCausalLM (Backbone)                │        │
│  │           Hidden dim: 896                            │        │
│  │           Causal attention                           │        │
│  │           Output: (batch, seq_len, 896)             │        │
│  └──────────────────────┬──────────────────────────────┘        │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │           LLM Decoder Head                           │        │
│  │           Linear(896 → 6761)                         │        │
│  │           Logits → Top-K(25) + Top-P(0.8) sampling  │        │
│  └──────────────────────┬──────────────────────────────┘        │
│                         ▼                                        │
│                   Speech Token (0-6560)                           │
│                   or EOS (6561)                                   │
│                                                                  │
│  Token Space:                                                    │
│  ┌────────────┬──────┬──────┬─────────┬──────────┐              │
│  │ 0 — 6560   │ 6561 │ 6562 │  6563   │6564-6760 │              │
│  │ Speech FSQ │ EOS  │ SOS  │ TaskID  │ Reserved │              │
│  └────────────┴──────┴──────┴─────────┴──────────┘              │
│                                                                  │
│  Silent Tokens (suppressed after 5 consecutive):                 │
│  [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]          │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Flow Decoder: CausalMaskedDiffWithDiT

```
┌──────────────────────────────────────────────────────────────────────┐
│  CausalMaskedDiffWithDiT                                              │
│                                                                       │
│  Speech Tokens (1, T_tok)                                            │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────────────────────┐                                        │
│  │  input_embedding          │  nn.Embedding(6561, 512)              │
│  │  (1, T_tok) → (1, T_tok, 512)                                    │
│  └────────────┬─────────────┘                                        │
│               ▼                                                       │
│  ┌──────────────────────────┐                                        │
│  │  PreLookaheadLayer        │  ConvNeXtV2 blocks                    │
│  │  Causal with 3-token      │  Lookahead: context=tokens[-3:]       │
│  │  lookahead                │  Output: (1, T_tok, 80)               │
│  └────────────┬─────────────┘                                        │
│               ▼                                                       │
│  ┌──────────────────────────┐                                        │
│  │  repeat_interleave(×2)    │  token_mel_ratio = 2                  │
│  │  (1, T_tok, 80)           │  1 token → 2 mel frames              │
│  │  → (1, T_mel, 80)         │  T_mel = T_tok × 2                   │
│  └────────────┬─────────────┘                                        │
│               ▼                                                       │
│  Speaker Embedding: Linear(192 → 80)                                 │
│               │                                                       │
│               ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  CausalConditionalCFM (Flow Matching Decoder)                 │    │
│  │                                                               │    │
│  │  Inputs:                                                      │    │
│  │    mu:   (1, 80, T_mel)  — encoded token features             │    │
│  │    cond: (1, 80, T_mel)  — prompt mel + zeros                 │    │
│  │    spks: (1, 80)         — speaker embedding                  │    │
│  │    mask: (1, 1, T_mel)   — padding mask                       │    │
│  │                                                               │    │
│  │  ┌─────────────────────────────────────────────────────┐      │    │
│  │  │  Euler ODE Solver (10 steps, cosine schedule)        │      │    │
│  │  │                                                      │      │    │
│  │  │  z ~ N(0, I)  ───────────────────────────► x(t=1)   │      │    │
│  │  │  t=0            step 1  step 2  ...  step 10         │      │    │
│  │  │                                                      │      │    │
│  │  │  Per step: CFG with batch=2                          │      │    │
│  │  │  ┌─────────┐  ┌─────────┐                           │      │    │
│  │  │  │Conditional│ │Uncondit. │  → blend with cfg=0.7    │      │    │
│  │  │  │ (mu,spks) │ │(zeros)   │                          │      │    │
│  │  │  └─────┬─────┘ └────┬────┘                           │      │    │
│  │  │        └──────┬──────┘                               │      │    │
│  │  │               ▼                                      │      │    │
│  │  │  x += dt × ((1+0.7)×v_cond - 0.7×v_uncond)         │      │    │
│  │  └─────────────────────────────────────────────────────┘      │    │
│  │                                                               │    │
│  │  Output: mel spectrogram (1, 80, T_mel) float32               │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 4. DiT Estimator (22-Layer Diffusion Transformer)

```
┌─────────────────────────────────────────────────────────────────┐
│  DiT (Diffusion Transformer)                                     │
│  dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2           │
│                                                                  │
│  Inputs (all transposed to seq-first):                           │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌───┐                             │
│  │ x  │ │cond│ │ mu │ │spks│ │ t │                              │
│  │80  │ │ 80 │ │ 80 │ │ 80 │ │ 1 │                              │
│  └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘ └─┬─┘                             │
│     └───┬───┘──────┘──────┘     │                                │
│         ▼                       ▼                                │
│  ┌──────────────┐   ┌────────────────────┐                      │
│  │InputEmbedding│   │ TimestepEmbedding  │                      │
│  │cat→Linear    │   │ sinusoidal→Linear  │                      │
│  │320 → 1024    │   │ 1 → 1024           │                      │
│  └──────┬───────┘   └────────┬───────────┘                      │
│         │                    │                                   │
│         ▼                    │                                   │
│  ┌──────────────────────┐    │                                   │
│  │CausalConvPosEmbed    │    │                                   │
│  │ (causal conv1d)      │    │                                   │
│  └──────────┬───────────┘    │                                   │
│             ▼                ▼                                   │
│  ╔══════════════════════════════════════╗  ×22 layers            │
│  ║  DiTBlock                            ║                        │
│  ║  ┌──────────────────────────────┐   ║                        │
│  ║  │ AdaLayerNormZero (time cond) │   ║                        │
│  ║  │ t_emb → γ, β, α modulation  │   ║                        │
│  ║  └──────────────┬───────────────┘   ║                        │
│  ║                 ▼                    ║                        │
│  ║  ┌──────────────────────────────┐   ║                        │
│  ║  │ Multi-Head Self-Attention     │   ║                        │
│  ║  │ heads=16, dim_head=64         │   ║                        │
│  ║  │ RoPE positional encoding      │   ║                        │
│  ║  │ Causal mask (chunk_size=50)   │   ║                        │
│  ║  └──────────────┬───────────────┘   ║                        │
│  ║                 ▼                    ║                        │
│  ║  ┌──────────────────────────────┐   ║                        │
│  ║  │ AdaLayerNormZero (time cond) │   ║                        │
│  ║  └──────────────┬───────────────┘   ║                        │
│  ║                 ▼                    ║                        │
│  ║  ┌──────────────────────────────┐   ║                        │
│  ║  │ FeedForward                   │   ║                        │
│  ║  │ 1024 → 2048 → 1024           │   ║                        │
│  ║  │ GELU activation              │   ║                        │
│  ║  └──────────────┬───────────────┘   ║                        │
│  ╚══════════════════╪══════════════════╝                        │
│                     ▼                                            │
│  ┌──────────────────────────────────┐                           │
│  │ AdaLayerNormZero_Final           │                           │
│  │ Final time-conditioned norm      │                           │
│  └──────────────┬───────────────────┘                           │
│                 ▼                                                │
│  ┌──────────────────────────────────┐                           │
│  │ Output Projection                │                           │
│  │ Linear(1024 → 80)               │                           │
│  └──────────────┬───────────────────┘                           │
│                 ▼                                                │
│         (batch, T_mel, 80) → transpose → (batch, 80, T_mel)    │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Vocoder: CausalHiFTGenerator

```
┌──────────────────────────────────────────────────────────────────┐
│  CausalHiFTGenerator                                              │
│  Mel (80-dim, 50Hz) → Waveform (24kHz)                           │
│                                                                   │
│  Mel Input (1, 80, T_mel)                                        │
│       │                                                           │
│       ├──────────────────────────────────┐                       │
│       ▼                                  ▼                        │
│  ┌──────────────┐              ┌──────────────────┐              │
│  │ F0 Predictor │              │ Main Conv Path    │              │
│  │ (on CPU)     │              │                   │              │
│  │ 5× Conv1d    │              │ Conv1d(80, 512)   │              │
│  │ 80→512→1     │              │ LeakyReLU(0.1)    │              │
│  │ → F0 contour │              └────────┬──────────┘              │
│  └──────┬───────┘                       │                        │
│         ▼                               │                        │
│  ┌──────────────┐                       │                        │
│  │ SineGen2     │                       │                        │
│  │ 8 harmonics  │                       │                        │
│  │ + noise      │                       │                        │
│  │ → source     │                       │                        │
│  └──────┬───────┘                       │                        │
│         │                               │                        │
│         ▼                               ▼                        │
│  ┌──────────────────────────────────────────────────┐           │
│  │  Upsample Block 1: ×8                             │           │
│  │  ConvTranspose1d(512, 256, k=16, s=8)            │           │
│  │  + 3× ResBlock (dilations [1,3,5])               │           │
│  │  + Source signal integration (NSF α=0.1)         │           │
│  │  50 Hz → 400 Hz                                   │           │
│  ├───────────────────────────────────────────────────┤           │
│  │  Upsample Block 2: ×5                             │           │
│  │  ConvTranspose1d(256, 128, k=11, s=5)            │           │
│  │  + 3× ResBlock (dilations [1,3,5])               │           │
│  │  + Source signal integration                      │           │
│  │  400 Hz → 2000 Hz                                 │           │
│  ├───────────────────────────────────────────────────┤           │
│  │  Upsample Block 3: ×3                             │           │
│  │  ConvTranspose1d(128, 64, k=7, s=3)              │           │
│  │  + 3× ResBlock (dilations [1,3,5])               │           │
│  │  + Source signal integration                      │           │
│  │  2000 Hz → 6000 Hz                                │           │
│  ├───────────────────────────────────────────────────┤           │
│  │  iSTFT Synthesis                                  │           │
│  │  Conv1d → magnitude + phase                       │           │
│  │  iSTFT(n_fft=16, hop_len=4)                      │           │
│  │  6000 Hz × 4 = 24000 Hz                          │           │
│  └───────────────────────────────────────────────────┘           │
│         │                                                        │
│         ▼                                                        │
│  Waveform: (1, T_samples) clipped to ±0.99                      │
│  T_samples = T_mel × 480 (total upsample: 8×5×3×4 = 480)       │
└──────────────────────────────────────────────────────────────────┘
```

## Streaming Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Streaming Inference (CosyVoice3Model.tts with stream=True)              │
│                                                                          │
│  ┌─────────────────────┐         ┌──────────────────────────────────┐   │
│  │ LLM Thread           │         │ Main Thread                      │   │
│  │                      │  tokens │                                  │   │
│  │ Generate tokens      ├────────►│ Poll token_event (10ms)          │   │
│  │ one at a time        │  event  │                                  │   │
│  │ → silent filtering   │         │ Accumulate tokens                │   │
│  │ → append to dict     │         │                                  │   │
│  │                      │         │ When enough tokens:              │   │
│  └──────────────────────┘         │  ┌─────────────────────────┐    │   │
│                                    │  │ token2wav()             │    │   │
│  Token Accumulation:               │  │  ├── Flow inference     │    │   │
│  ┌───────────────────────────┐    │  │  ├── Mel clamp [-20,2]  │    │   │
│  │ Chunk 1: 25 tokens        │    │  │  ├── Mel cache append   │    │   │
│  │ Chunk 2: 25 tokens        │    │  │  ├── HiFiGAN inference  │    │   │
│  │ Chunk 3: 100 tokens (max) │    │  │  └── Yield audio chunk  │    │   │
│  │ ...                       │    │  └─────────────────────────┘    │   │
│  └───────────────────────────┘    │                                  │   │
│                                    │ CUDA Stream Pool (8 streams)     │   │
│  Hop growth: 25 → 100 → 100...    │ for concurrent token2wav         │   │
│  (×4 scale, capped at 100)        └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

## Tensor Shape Flow (Concrete Example: 1 second of audio)

```
"Hello world"
     │
     ▼
Tokenized text:          (1, 12)         int32       Qwen2 token IDs
     │
     ▼ LLM generates 25 speech tokens
Speech tokens:           (1, 25)         int32       FSQ v3 (0-6560)
     │
     ▼ input_embedding
Token embeddings:        (1, 25, 512)    float32
     │
     ▼ PreLookaheadLayer
Encoded features:        (1, 25, 80)     float32
     │
     ▼ repeat_interleave(×2)
Mel-rate features:       (1, 50, 80)     float32     token_mel_ratio=2
     │
     ▼ transpose → (1, 80, 50)
     │
     ▼ DiT decoder (10 Euler steps × batch=2 for CFG)
     │   Per step: DiT forward on (2, 50, 1024) through 22 layers
     │
Mel spectrogram:         (1, 80, 50)     float32     50 frames @ 50Hz = 1s
     │
     ▼ HiFiGAN (upsample ×480)
Waveform:                (1, 24000)      float32     24000 samples @ 24kHz = 1s
```

## Key Dimensions Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 24,000 Hz | Output audio |
| Mel bins | 80 | Spectrogram dimension |
| Mel hop size | 480 samples | 20ms @ 24kHz → 50 Hz mel |
| Mel FFT size | 1,920 | Window size |
| Token frame rate | 25 Hz | 1 token = 40ms |
| Token-mel ratio | 2 | 1 token → 2 mel frames |
| FSQ vocab size | 6,561 | Speech token vocabulary |
| Speaker embed dim | 192 | CampPlus output |
| LLM hidden dim | 896 | Qwen2 backbone |
| DiT hidden dim | 1,024 | Transformer width |
| DiT layers | 22 | Transformer depth |
| DiT attention heads | 16 | 64-dim per head |
| DiT FFN dim | 2,048 | ff_mult=2 |
| DiT chunk size | 50 | Streaming causal window |
| Vocoder base channels | 512 | HiFiGAN width |
| Vocoder upsample | 8×5×3×4=480 | Mel frame → samples |
| Diffusion steps | 10 | Euler ODE solver |
| CFG rate (inference) | 0.7 | Guidance strength |
| CFG rate (training) | 0.2 | Condition dropout |

## Compute Budget Per 1 Second of Audio

| Component | Forward Passes | Tensor Size | Bottleneck |
|-----------|---------------|-------------|------------|
| **LLM** | 25 autoregressive steps | (1, ~37, 896) | Memory bandwidth |
| **DiT** | 10 steps × 2 (CFG) × 22 layers = **440** | (2, 50, 1024) | **Compute bound** |
| **Vocoder** | 1 forward pass | (1, 80, 50) → (1, 24000) | Lightweight |

The DiT is the clear bottleneck: 440 layer forward passes per second of audio.
