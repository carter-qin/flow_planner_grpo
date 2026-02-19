#!/bin/bash
# 确保在 flow_planner_grpo 根目录下运行

# bash data_process.sh

python generate_json.py

# 1. 执行训练
echo ">>> [$(date)] Starting Training..."
cd flow_planner/run_script/
bash launch_train.sh

# 检查训练返回值 (0 代表正常结束)
if [ $? -eq 0 ]; then
    echo ">>> [$(date)] Training completed successfully. Starting Simulation..."
    
    # 2. 返回根目录并执行仿真
    cd ../..
    bash flow_planner/run_script/launch_sim_nuplan_batch.sh
else
    echo ">>> [$(date)] Training failed with exit code $?. Skipping simulation."
    exit 1
fi

echo ">>> [$(date)] All tasks finished."