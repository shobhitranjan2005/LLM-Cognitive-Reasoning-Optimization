#!/bin/bash
K="shobhitranjan2005/loom-v2-finetune-sweep"
for i in $(seq 1 55); do
  S=$(kaggle kernels status "$K" 2>&1 | tail -1)
  echo "[$(date +%H:%M:%S)] $S"
  case "$S" in
    *RUNNING*|*QUEUED*) sleep 300 ;;
    *) echo "FINAL: $S"; exit 0 ;;
  esac
done
echo "still running after ~4.5h"; exit 1
