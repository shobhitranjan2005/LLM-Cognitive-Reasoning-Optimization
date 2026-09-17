#!/bin/bash
# Poll the Kaggle kernel until it stops running, then exit so the session is notified.
K="shobhitranjan2005/loom-v2-finetune-sweep"
for i in $(seq 1 60); do          # 60 x 5 min = 5 h ceiling
  S=$(kaggle kernels status "$K" 2>&1 | tail -1)
  echo "[$(date +%H:%M:%S)] $S"
  case "$S" in
    *RUNNING*|*QUEUED*) sleep 300 ;;
    *) echo "FINAL: $S"; exit 0 ;;
  esac
done
echo "gave up waiting after 5 h"; exit 1
