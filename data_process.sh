#!/bin/bash
set -e  # [建议] 任何一个任务出错立即停止，避免连锁反应

###################################
# 1. Mini Split
###################################
# echo ">>> Starting Mini Split..."
# NUPLAN_DATA_PATH="$HOME/data/nuplan/dataset/nuplan-v1.1/splits/mini"
NUPLAN_MAP_PATH="$HOME/data/nuplan/dataset/maps"
# TRAIN_SET_PATH="$HOME/data/nuplan/processed_data_mini"

# python data_multiprocess.py \
# --data_path $NUPLAN_DATA_PATH \
# --map_path $NUPLAN_MAP_PATH \
# --save_path $TRAIN_SET_PATH \
# --total_scenarios 1000000 \
# --num_workers 10 \
# --shuffle_scenarios 0

###################################
# 2. Train Pittsburgh
###################################
echo ">>> Starting Train Pittsburgh..."
NUPLAN_DATA_PATH="$HOME/data/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
# 地图路径不用变，可以复用上面的，或者重新赋值也行
TRAIN_SET_PATH="$HOME/data/nuplan/processed_data_train_pittsburgh"

python data_multiprocess.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \
--num_workers 12 \
--shuffle_scenarios 0

###################################
# 3. Validation
###################################
# echo ">>> Starting Validation..."
# NUPLAN_DATA_PATH="$HOME/data/nuplan/dataset/nuplan-v1.1/splits/val"
# VAL_SET_PATH="$HOME/data/nuplan/processed_data_val"

# python data_multiprocess.py \
# --data_path $NUPLAN_DATA_PATH \
# --map_path $NUPLAN_MAP_PATH \
# --save_path $VAL_SET_PATH \
# --total_scenarios 1000000 \
# --num_workers 12 \
# --shuffle_scenarios 0

echo ">>> All tasks finished successfully!"