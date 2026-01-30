#!/usr/bin/env bash
# 分布式训练启动脚本。
# 若报错 EADDRINUSE (address already in use)，请任选其一：
#   1) 换端口: MASTER_PORT=29501 ./run_train_ddp.sh
#   2) 结束占用: fuser -k 29500/tcp  或  kill $(lsof -t -i:29500)
# 用法:
#   ./run_train_ddp.sh
#   MASTER_PORT=29501 ./run_train_ddp.sh
#   NGPUS=4 ./run_train_ddp.sh

export MASTER_PORT="${MASTER_PORT:-29500}"
NGPUS="${NGPUS:-4}"

echo "Using MASTER_PORT=$MASTER_PORT NGPUS=$NGPUS"
torchrun --nproc_per_node="$NGPUS" train_ddp.py "$@"
