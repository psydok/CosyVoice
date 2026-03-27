#!/usr/bin/env bash
set -euo pipefail

LEN_TEXT="${LEN_TEXT:-100}"
THREADS_LIST="${THREADS_LIST:-1 2 4 8 16 32}"
NUM_INSTANCE=32

echo "ПРОГРЕВ..."
python3 client_grpc.py --log-dir "./log_with_cache_text${LEN_TEXT}_threads${NUM_INSTANCE}" --num-tasks "${NUM_INSTANCE}" --use-spk2info-cache False
echo "ПРОГРЕВ ЗАВЕРШЕН..."

for NUM_THREADS in ${THREADS_LIST}; do
  python3 client_grpc.py --log-dir "./log_with_cache_text${LEN_TEXT}_threads${NUM_THREADS}" --num-tasks "${NUM_THREADS}" --use-spk2info-cache True
done
echo "ТЕСТИРОВАНИЕ ЗАВЕРШЕНО..."

python3 benchmark_report.py
echo "ОТЧЕТ ГОТОВ"
