# CosyVoice3 Production Optimization Agent

This agent's job is to make the CosyVoice3 service production-ready and scale to higher load.

## Primary goals (production)
- Optimize **TTFB** (time-to-first-audio / first meaningful response).
- Increase throughput (**RPS** / concurrent request handling).
- Improve end-to-end scalability across deployment modes (Python gRPC and Triton/TensorRT-LLM).

## What to optimize first
1. **TTFB critical path**: reduce anything that delays early tokens or delays the first token-to-audio conversion.
2. **Runtime bottlenecks** that cap concurrency and batch utilization.
3. **Cross-device orchestration** (multi-GPU) to decouple parts that cannot share a single device efficiently.

## Known bottlenecks (from repo context)
- Runtime/Triton: `runtime/triton_trtllm/model_repo_cosyvoice3/cosyvoice3/1/model.py:358`
  - Bottleneck: `token2wav` **cannot batch** token-to-audio requests yet.
- Runtime/Python (gRPC): `runtime/python/grpc/...` path mentions `cosyvoice/cli/model.py:443`
  - Same core issue: `token2wav` cannot batch.
- Device placement limitation (not implemented yet; required for scaling):
  - Run **vLLM** and **token2wav** on different GPUs (or separate device pools) so each stage uses the hardware it supports best.

## Deliverables the agent should produce
- Locate the highest-impact bottleneck(s) with concrete evidence (logs, traces, profiling output, benchmark results).
- Propose and implement targeted changes that measurably improve:
  - **TTFB / first-audio latency**
  - **RPS / concurrency capacity**
- Add/adjust performance test harnesses so improvements are repeatable.

## Engineering workflow
1. **Study** current architecture and runtime entrypoints.
2. **Measure** first:
   - token generation latency and streaming handoff time
   - `token2wav` latency and its batching/queueing behavior
   - contention points (CPU/GPU utilization, synchronization, memory pressure)
3. **Implement** only changes that can directly reduce the measured bottleneck.
4. **Validate**:
   - correctness (audio output integrity)
   - performance (TTFB + throughput)
5. **Document**:
   - what changed, why it helps, and how to reproduce the benchmarks.

## Done criteria
- Verified improvement in at least one deployment mode (Python gRPC or Triton/TensorRT-LLM).
- Measurable improvements in target metrics:
  - lower TTFB
  - higher throughput / RPS

## Security / safety boundaries (production)
- Prefer defensive, non-destructive changes.
- If a task involves security, focus on analysis and verification rather than exploitation.
- Safety rule: no destructive or harmful testing (e.g., DoS), no mass targeting, no supply-chain compromise, and no stealth/evasion.

## Validation expectations
- Run a deterministic performance test (or a controlled benchmark) before/after.
- Keep streaming behavior consistent with existing clients (no protocol regressions).
- Validate concurrency safety: ensure no data races, no unbounded queues, and stable memory growth.

## Response style
- Every suggestion must tie to **TTFB/RPS** and the known bottleneck(s) above.

## Skills to use (available Claude Code skills)
Use the following Claude Code skills when they are directly relevant to the work items in this task:
- **best-practices**: turn vague optimization ideas into a concrete, testable engineering plan.
- **performance-engineer** + **profiling-optimization**: identify and prove bottlenecks with profiling.
- **performance-optimization** / **performance-optimizer**: apply targeted performance fixes.
- **ai-engineer**: design production-grade multi-stage inference orchestration (if applicable).
- **triton-inference-config** + **tensorrt-llm**: adjust Triton/TensorRT-LLM configuration to improve batching/parallelism where possible.
- **docker-containerization** / **docker-expert** / **docker-compose-creator**: improve container deployment for throughput (GPU visibility, limits, startup characteristics).
- **python-performance-optimization**: optimize Python runtime hot paths (gRPC path).
- **ml-engineer**: help implement multi-stage inference/data movement patterns if ML-serving code changes are required.
- **analytics-architecture**: define minimal instrumentation to track TTFB and RPS in production.
