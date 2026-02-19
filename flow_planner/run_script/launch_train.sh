export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=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
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT=$(cd ../.. && pwd)
export SAVE_DIR="$PROJECT_ROOT/work_dirs"
mkdir -p $SAVE_DIR
export TENSORBOARD_LOG_PATH="$SAVE_DIR/tensorboard_logs"
mkdir -p $TENSORBOARD_LOG_PATH
export TRAINING_DATA= # path to the training data npz
export TRAINING_JSON= # path to the training data list json
export TORCH_LOGS="dynamic,recompiles"

cleanup() {
    echo "Caught SIGINT, sending SIGKILL to all python processes..."
    pkill -P $$ # 杀掉当前脚本启动的所有子进程
    kill -9 $(jobs -p) 2>/dev/null
    exit 1
}

trap cleanup SIGINT SIGTERM

python -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone ../trainer.py \
    --config-name flow_planner_standard \
    ddp.world_size=1

pid=$!
wait $pid