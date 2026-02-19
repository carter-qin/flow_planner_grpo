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
###################################


BRANCH_NAME=flow_planner_rl
CONFIG_FILE="/root/data/flow_planner_grpo/work_dirs/outputs/mini/pittsburgh_10x_downsample_run1/2026-02-12_15-23-57/.hydra/config.yaml"
CKPT_FILE="/root/data/flow_planner_grpo/work_dirs/outputs/mini/pittsburgh_10x_downsample_run1/2026-02-12_15-23-57/latest.pth"

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=flow_planner

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.flow_planner.config_path=$CONFIG_FILE \
    planner.flow_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=sequential \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"