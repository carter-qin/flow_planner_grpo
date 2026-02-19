export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="$HOME/data/nuplan-devkit" 
export NUPLAN_DATA_ROOT="$HOME/data/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/data/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/data/nuplan"

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_reactive_agents"

# --- Batching 核心配置 ---
# 每次仿真处理的场景数量 (根据显存大小调整，50是个比较保守的安全值)
BATCH_SIZE=600
# 总共想要跑多少个场景 (如果想跑完 val14，这里建议设为 2500 或者更大)
TOTAL_SCENARIOS=1200 
###################################


BRANCH_NAME=flow_planner_rl
CONFIG_FILE="/root/data/flow_planner_grpo/work_dirs/outputs/pittsburgh_rl/pittsburgh_run3/.hydra/config.yaml"
CKPT_FILE="/root/data/flow_planner_grpo/work_dirs/outputs/pittsburgh_rl/pittsburgh_run3/latest.pth"

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=flow_planner

# --- 自动清理 Ray 缓存 (防止硬盘爆满) ---
echo ">>> Cleaning up Ray temp files..."
pkill -f ray
rm -rf /tmp/ray/*

# 定义每个 Batch 处理多少个
BATCH_SIZE=200
TOTAL_SCENARIOS=1200 # 或者是你估计的总数
NUM_BATCHES=$(( (TOTAL_SCENARIOS + BATCH_SIZE - 1) / BATCH_SIZE ))

# 计算批次总数
NUM_BATCHES=$(( (TOTAL_SCENARIOS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "=========================================================="
echo ">>> Starting Batched Inference for Flow Planner"
echo ">>> Checkpoint: $FILENAME"
echo ">>> Total Scenarios: $TOTAL_SCENARIOS"
echo ">>> Batch Size:      $BATCH_SIZE"
echo ">>> Total Batches:   $NUM_BATCHES"
echo "=========================================================="

for (( batch=0; batch<$NUM_BATCHES; batch++ ))
do
    START_IDX=$(( batch * BATCH_SIZE ))
    CURRENT_BATCH_NUM=$((batch+1))
    
    echo " "
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Running Batch $CURRENT_BATCH_NUM / $NUM_BATCHES (Offset: $START_IDX, Limit: $BATCH_SIZE)"
    echo "----------------------------------------------------------"

    # 执行 Python 脚本
    # 关键参数解释：
    # 1. +scenario_filter_offset=$START_IDX : 这是我们修改源码后新增的参数，必须带 '+' 号因为它是新增的 hydra key
    # 2. scenario_filter.limit_total_scenarios : 限制当前进程只跑 N 个
    # 3. scenario_filter.shuffle=false : 必须关闭 shuffle，保证切片的确定性（不会重复跑同一个场景）
    # 4. worker=sequential : 单进程顺序执行，最大程度节省内存
    
    python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
        +simulation=$CHALLENGE \
        planner=$PLANNER \
        planner.flow_planner.config_path=$CONFIG_FILE \
        planner.flow_planner.ckpt_path=$CKPT_FILE \
        planner.flow_planner.enable_lora=true \
        scenario_builder=$SCENARIO_BUILDER \
        scenario_filter=$SPLIT \
        scenario_filter.shuffle=false \
        scenario_filter.limit_total_scenarios=$BATCH_SIZE \
        +scenario_filter_offset=$START_IDX \
        experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_batch${batch} \
        verbose=true \
        worker=ray_distributed \
        worker.threads_per_node=10 \
        distributed_mode='SINGLE_NODE' \
        number_of_gpus_allocated_per_simulation=0.3 \
        number_of_cpus_allocated_per_simulation=3 \
        enable_simulation_progress_bar=true \
        hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
    
    # 错误检测
    EXIT_CODE=$?

    echo ">>> [Cleanup] Wiping Ray temp files for next batch..."
    # 强制停止可能残留的 Ray 进程
    ray stop --force > /dev/null 2>&1
    pkill -f ray
    # 物理删除临时文件
    rm -rf /tmp/ray/*

    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Batch $CURRENT_BATCH_NUM failed with exit code $EXIT_CODE! Stopping sequence."
        # 如果是 OOM 或代码错误，建议直接退出，避免浪费时间
        exit 1 
    else
        echo "✅ Batch $CURRENT_BATCH_NUM completed successfully."
    fi

    # 显存/内存冷却 & 垃圾回收
    # 对于 Python 这种引用计数语言，进程结束后 OS 会回收所有资源，sleep 主要是为了防止 io 拥堵或日志写入冲突
    sleep 3
done

echo "=========================================================="
echo ">>> All batches completed."
echo "=========================================================="