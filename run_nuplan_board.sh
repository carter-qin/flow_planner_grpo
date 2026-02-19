#!/bin/bash

# 1. 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate flow_planner

# 2. 设置路径
export NUPLAN_DEVKIT_ROOT="/root/data/nuplan-devkit"
export NUPLAN_DATA_ROOT="/root/data/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/root/data/nuplan/dataset/maps"

# ===== 配置要对比的实验路径 =====
export BASE_PATH="/root/data/nuplan/exp/simulation/closed_loop_reactive_agents/flow_planner/val14"

export CHALLENGE="closed_loop_reactive_agents"

# 要对比的实验名称列表
EXPERIMENTS=(
    "flow_planner_lora"
    "flow_planner_rl"
)

export EXP_LIST=$(IFS=,; echo "${EXPERIMENTS[*]}")

echo "=================================================="
echo ">>> 1. Merging batches for each experiment..."
echo "=================================================="

for EXP_NAME in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_PATH="$BASE_PATH/$EXP_NAME"
    MERGED_PATH="$EXPERIMENT_PATH/merged"
    
    echo ""
    echo "  Processing: [$EXP_NAME]"
    echo "  Source: $EXPERIMENT_PATH"
    echo "  Merged: $MERGED_PATH"
    
    # 清理旧的 merged
    rm -rf "$MERGED_PATH"
    
    # 创建 merged 目录结构
    mkdir -p "$MERGED_PATH/metrics"
    
    # -------------------------------------------------------
    # 合并 simulation_log:
    # 目录结构: simulation_log/planner_name/scenario_type/scenario_name/
    # -------------------------------------------------------
    echo "    Merging simulation_log..."
    TOTAL_SIM=0
    for BATCH_DIR in "$EXPERIMENT_PATH"/latest_batch*; do
        SIM_LOG="$BATCH_DIR/simulation_log"
        if [ ! -d "$SIM_LOG" ]; then
            continue
        fi
        for PLANNER_DIR in "$SIM_LOG"/*/; do
            PLANNER_NAME=$(basename "$PLANNER_DIR")
            mkdir -p "$MERGED_PATH/simulation_log/$PLANNER_NAME"
            for STYPE_DIR in "$PLANNER_DIR"/*/; do
                STYPE_NAME=$(basename "$STYPE_DIR")
                mkdir -p "$MERGED_PATH/simulation_log/$PLANNER_NAME/$STYPE_NAME"
                for SCENARIO_DIR in "$STYPE_DIR"/*/; do
                    [ -d "$SCENARIO_DIR" ] || continue
                    SCENARIO_NAME=$(basename "$SCENARIO_DIR")
                    TARGET="$MERGED_PATH/simulation_log/$PLANNER_NAME/$STYPE_NAME/$SCENARIO_NAME"
                    if [ ! -e "$TARGET" ]; then
                        ln -s "$(realpath "$SCENARIO_DIR")" "$TARGET"
                        TOTAL_SIM=$((TOTAL_SIM + 1))
                    fi
                done
            done
        done
    done
    echo "    -> simulation_log: $TOTAL_SIM scenarios linked"
done

echo ""
echo "=================================================="
echo ">>> 2. Re-building aggregator_metric, metrics & .nuboard files..."
echo "=================================================="

python - <<'PYEOF'
import os, sys, pickle, re, glob
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from collections import defaultdict, OrderedDict
from copy import deepcopy

sys.path.insert(0, os.environ.get("NUPLAN_DEVKIT_ROOT", ""))

base_path = Path(os.environ.get("BASE_PATH", ""))
experiments = os.environ.get("EXP_LIST", "").split(",")
challenge = os.environ.get("CHALLENGE", "closed_loop_reactive_agents")
devkit_root = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", ""))
nuboard_paths = []

# ===========================================================
# Load metric_weights and multiple_metrics from NuPlan devkit config
# ===========================================================
aggregator_config_path = (
    devkit_root / "nuplan" / "planning" / "script" / "config" / "simulation"
    / "metric_aggregator" / f"{challenge}_weighted_average.yaml"
)

metric_weights = {'default': 1.0}
multiple_metrics = []

if aggregator_config_path.exists():
    with open(aggregator_config_path, 'r') as f:
        agg_cfg = yaml.safe_load(f)
    
    # YAML structure: {challenge}_weighted_average: { metric_weights: {...}, multiple_metrics: [...] }
    top_key = f"{challenge}_weighted_average"
    wa_cfg = agg_cfg.get(top_key, agg_cfg)
    
    if 'metric_weights' in wa_cfg and wa_cfg['metric_weights']:
        metric_weights = wa_cfg['metric_weights']
    if 'multiple_metrics' in wa_cfg and wa_cfg['multiple_metrics']:
        multiple_metrics = wa_cfg['multiple_metrics']
    
    print(f"  Loaded aggregator config: {aggregator_config_path}")
else:
    print(f"  [WARN] Config not found: {aggregator_config_path}")

print(f"  metric_weights: {metric_weights}")
print(f"  multiple_metrics: {multiple_metrics}")

for exp_name in experiments:
    exp_name = exp_name.strip()
    if not exp_name:
        continue
    
    exp_path = base_path / exp_name
    merged_path = exp_path / "merged"
    merged_agg = merged_path / "aggregator_metric"
    merged_agg.mkdir(parents=True, exist_ok=True)
    merged_metrics = merged_path / "metrics"
    merged_metrics.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  === Processing [{exp_name}] ===")
    
    batch_dirs = sorted(glob.glob(str(exp_path / "latest_batch*")))
    print(f"  Found {len(batch_dirs)} batch directories")
    
    # ===========================================================
    # Step 2a: Merge per-metric parquets from all batches
    # ===========================================================
    metric_files_by_name = defaultdict(list)
    for batch_dir in batch_dirs:
        metrics_dir = Path(batch_dir) / "metrics"
        if not metrics_dir.exists():
            continue
        for pq_file in metrics_dir.glob("*.parquet"):
            metric_files_by_name[pq_file.name].append(pq_file)
    
    print(f"  Found {len(metric_files_by_name)} distinct metric types")
    
    for metric_filename, pq_files in sorted(metric_files_by_name.items()):
        dfs = []
        for pq_file in pq_files:
            try:
                df = pd.read_parquet(pq_file)
                dfs.append(df)
            except Exception as e:
                print(f"    [WARN] Failed to read {pq_file}: {e}")
        
        if not dfs:
            continue
        
        combined = pd.concat(dfs, ignore_index=True)
        
        if 'scenario_name' in combined.columns:
            before = len(combined)
            combined = combined.drop_duplicates(subset='scenario_name', keep='first')
            if before != len(combined):
                print(f"    {metric_filename}: dedup {before} -> {len(combined)} rows")
        
        out_path = merged_metrics / metric_filename
        combined.to_parquet(out_path)
        print(f"    Merged {metric_filename}: {len(pq_files)} files -> {len(combined)} rows")
    
    # ===========================================================
    # Step 2b: Re-run the aggregator logic from scratch
    # Reference: weighted_average_metric_aggregator.py
    # ===========================================================
    metric_dataframes = {}
    for pq_file in sorted(merged_metrics.glob("*.parquet")):
        metric_name = pq_file.stem
        try:
            df = pd.read_parquet(pq_file)
            metric_dataframes[metric_name] = df
        except Exception as e:
            print(f"    [WARN] Failed to read {pq_file}: {e}")
    
    print(f"  Loaded {len(metric_dataframes)} metric dataframes for aggregation")
    
    if not metric_dataframes:
        print(f"  [ERROR] No metric dataframes for {exp_name}")
        continue
    
    # ---- Get planner name ----
    planner_names = set()
    for metric_name, df in metric_dataframes.items():
        if 'planner_name' in df.columns:
            planner_names.update(df['planner_name'].unique())
    planner_names = sorted(planner_names)
    print(f"  Planner names: {planner_names}")
    
    metric_names_sorted = sorted(metric_dataframes.keys())
    
    for planner_name in planner_names:
        print(f"\n  --- Planner: {planner_name} ---")
        
        # ---- Step A: _group_scenario_metrics ----
        columns_template = OrderedDict()
        for col in ['log_name', 'planner_name', 'aggregator_type', 'scenario_type', 'num_scenarios']:
            columns_template[col] = None
        for mn in metric_names_sorted:
            columns_template[mn] = None
        columns_template['score'] = None
        
        scenario_metric_columns = OrderedDict()
        
        for metric_name, df in metric_dataframes.items():
            if 'planner_name' in df.columns:
                planner_df = df[df['planner_name'] == planner_name]
            else:
                planner_df = df
            
            for _, data in planner_df.iterrows():
                scenario_name = data.get('scenario_name')
                if scenario_name is None:
                    continue
                
                if scenario_name not in scenario_metric_columns:
                    scenario_metric_columns[scenario_name] = deepcopy(columns_template)
                
                scenario_metric_columns[scenario_name]['log_name'] = data.get('log_name')
                scenario_metric_columns[scenario_name]['planner_name'] = planner_name
                scenario_metric_columns[scenario_name]['scenario_type'] = data.get('scenario_type')
                scenario_metric_columns[scenario_name]['aggregator_type'] = 'weighted_average'
                scenario_metric_columns[scenario_name][metric_name] = data.get('metric_score')
        
        total_scenarios = len(scenario_metric_columns)
        print(f"    Total scenarios: {total_scenarios}")
        
        # ---- Step B: _compute_scenario_score ----
        # score = multiple_factor * weighted_avg(other scored metrics)
        # multiple_factor = product of multiple_metrics scores
        # weighted_avg = sum(weight_i * score_i) / sum(weight_i)
        excluded_columns = {'log_name', 'planner_name', 'aggregator_type', 'scenario_type', 'num_scenarios', 'score'}
        
        for scenario_name, columns in scenario_metric_columns.items():
            metric_scores = 0.0
            sum_weights = 0.0
            multiple_factor = 1.0
            
            for column_key, column_value in columns.items():
                if column_key in excluded_columns or column_value is None:
                    continue
                if column_key in multiple_metrics:
                    multiple_factor *= column_value
                else:
                    weight = metric_weights.get(column_key, metric_weights.get('default', 1.0))
                    sum_weights += weight
                    metric_scores += weight * column_value
            
            weighted_average_score = metric_scores / sum_weights if sum_weights else 0.0
            final_score = multiple_factor * weighted_average_score
            scenario_metric_columns[scenario_name]['score'] = final_score
        
        # ---- Step C: _group_scenario_type_metric ----
        scenario_type_dicts = defaultdict(lambda: defaultdict(list))
        for scenario_name, columns in scenario_metric_columns.items():
            scenario_type = columns['scenario_type']
            scenario_type_dicts[scenario_type]['scenario_name'].append(scenario_name)
            for column_key, column_value in columns.items():
                scenario_type_dicts[scenario_type][column_key].append(column_value)
        
        common_columns = {'planner_name', 'aggregator_type', 'scenario_type'}
        excluded_group_columns = {'scenario_name'}
        
        scenario_type_metric_columns = OrderedDict()
        for scenario_type, columns in sorted(scenario_type_dicts.items()):
            scenario_type_metric_columns[scenario_type] = OrderedDict()
            for key, values in columns.items():
                if key in excluded_group_columns:
                    continue
                elif key in common_columns:
                    scenario_type_metric_columns[scenario_type][key] = values[0]
                elif key == 'log_name':
                    scenario_type_metric_columns[scenario_type][key] = None
                elif key == 'num_scenarios':
                    scenario_type_metric_columns[scenario_type]['num_scenarios'] = len(values)
                else:
                    available_values = np.asarray([v for v in values if v is not None], dtype=np.float64)
                    if available_values.size > 0:
                        value = float(np.sum(available_values))
                    else:
                        value = None
                    
                    if key == 'score' and value is not None:
                        score_value = value / len(values) if len(values) else 0.0
                        scenario_type_metric_columns[scenario_type][key] = score_value
                    else:
                        scenario_type_metric_columns[scenario_type][key] = value
        
        # ---- Step D: _group_final_score_metric ----
        final_score_dicts = defaultdict(lambda: defaultdict(list))
        for scenario_type, columns in scenario_type_metric_columns.items():
            for column_key, column_value in columns.items():
                final_score_dicts['final_score'][column_key].append(column_value)
        
        final_total_scenarios = sum(final_score_dicts['final_score']['num_scenarios'])
        
        final_score_metric_columns = OrderedDict()
        final_score_metric_columns['final_score'] = OrderedDict()
        
        for key, values in final_score_dicts['final_score'].items():
            if key == 'scenario_type':
                final_score_metric_columns['final_score'][key] = 'final_score'
            elif key == 'log_name':
                final_score_metric_columns['final_score'][key] = None
            elif key in common_columns:
                final_score_metric_columns['final_score'][key] = values[0]
            elif key == 'num_scenarios':
                final_score_metric_columns['final_score'][key] = final_total_scenarios
            else:
                if key == 'score':
                    available_values = []
                    for value, num_scenario in zip(values, final_score_dicts['final_score']['num_scenarios']):
                        if value is not None:
                            available_values.append(value * num_scenario)
                else:
                    available_values = [v for v in values if v is not None]
                
                if not available_values:
                    total_values = None
                else:
                    available_value_array = np.asarray(available_values, dtype=np.float64)
                    total_values = float(np.sum(available_value_array)) / final_total_scenarios
                
                final_score_metric_columns['final_score'][key] = total_values
        
        # ---- Step E: Combine all into dataframe ----
        dataframe_columns = OrderedDict()
        dataframe_columns['scenario'] = []
        dataframe_columns['log_name'] = []
        dataframe_columns['scenario_type'] = []
        dataframe_columns['num_scenarios'] = []
        dataframe_columns['planner_name'] = []
        dataframe_columns['aggregator_type'] = []
        for mn in metric_names_sorted:
            dataframe_columns[mn] = []
        dataframe_columns['score'] = []
        
        for scenario_name, columns in scenario_metric_columns.items():
            dataframe_columns['scenario'].append(scenario_name)
            for key in list(dataframe_columns.keys()):
                if key == 'scenario':
                    continue
                dataframe_columns[key].append(columns.get(key))
        
        for scenario_type, columns in scenario_type_metric_columns.items():
            dataframe_columns['scenario'].append(scenario_type)
            for key in list(dataframe_columns.keys()):
                if key == 'scenario':
                    continue
                dataframe_columns[key].append(columns.get(key))
        
        for final_name, columns in final_score_metric_columns.items():
            dataframe_columns['scenario'].append(final_name)
            for key in list(dataframe_columns.keys()):
                if key == 'scenario':
                    continue
                dataframe_columns[key].append(columns.get(key))
        
        aggregated_df = pd.DataFrame(data=dataframe_columns)
        
        out_name = f"closed_loop_reactive_agents_weighted_average_metrics_merged.parquet"
        out_path = merged_agg / out_name
        aggregated_df.to_parquet(out_path)
        print(f"\n    Saved merged aggregator: {out_path} ({len(aggregated_df)} rows)")
        
        # ---- Preview summary rows ----
        summary = aggregated_df[aggregated_df['num_scenarios'].notna()]
        print(f"    Summary rows ({len(summary)}):")
        for _, row in summary.iterrows():
            stype = row['scenario_type']
            n = int(row['num_scenarios'])
            score = row['score']
            print(f"      {stype:>55s}  n={n:>5d}  score={score:.6f}")
        
        # ---- Verification: re-aggregate batch0 only, compare with original ----
        print(f"\n    === Verification: re-aggregate batch0 only ===")
        batch0_dir = Path(batch_dirs[0])
        b0_metrics_dir = batch0_dir / "metrics"
        b0_agg_dir = batch0_dir / "aggregator_metric"
        
        if b0_metrics_dir.exists() and b0_agg_dir.exists():
            b0_metric_dfs = {}
            for pq_file in sorted(b0_metrics_dir.glob("*.parquet")):
                b0_metric_dfs[pq_file.stem] = pd.read_parquet(pq_file)
            
            b0_scenario_cols = OrderedDict()
            for mn, df in b0_metric_dfs.items():
                pdf = df[df['planner_name'] == planner_name] if 'planner_name' in df.columns else df
                for _, data in pdf.iterrows():
                    sn = data.get('scenario_name')
                    if sn is None:
                        continue
                    if sn not in b0_scenario_cols:
                        b0_scenario_cols[sn] = deepcopy(columns_template)
                    b0_scenario_cols[sn]['log_name'] = data.get('log_name')
                    b0_scenario_cols[sn]['planner_name'] = planner_name
                    b0_scenario_cols[sn]['scenario_type'] = data.get('scenario_type')
                    b0_scenario_cols[sn]['aggregator_type'] = 'weighted_average'
                    b0_scenario_cols[sn][mn] = data.get('metric_score')
            
            # Compute batch0 per-scenario scores with correct weights + multipliers
            for sn, cols in b0_scenario_cols.items():
                ms = 0.0; sw = 0.0; mf = 1.0
                for k, v in cols.items():
                    if k in excluded_columns or v is None:
                        continue
                    if k in multiple_metrics:
                        mf *= v
                    else:
                        w = metric_weights.get(k, metric_weights.get('default', 1.0))
                        sw += w; ms += w * v
                cols['score'] = mf * (ms / sw if sw else 0.0)
            
            # Group by type
            b0_type_dicts = defaultdict(lambda: defaultdict(list))
            for sn, cols in b0_scenario_cols.items():
                st = cols['scenario_type']
                for k, v in cols.items():
                    b0_type_dicts[st][k].append(v)
            
            b0_type_scores = {}
            for st, cols in sorted(b0_type_dicts.items()):
                n = len(cols['score'])
                s = sum(v for v in cols['score'] if v is not None) / n if n else 0.0
                b0_type_scores[st] = (n, s)
            
            ws = sum(n * s for st, (n, s) in b0_type_scores.items())
            wt = sum(n for st, (n, s) in b0_type_scores.items())
            b0_our_final = ws / wt if wt else 0.0
            
            b0_orig = pd.read_parquet(list(b0_agg_dir.glob("*.parquet"))[0])
            b0_orig_summary = b0_orig[b0_orig['num_scenarios'].notna()]
            b0_orig_final = b0_orig_summary[b0_orig_summary['scenario_type'] == 'final_score']
            
            if len(b0_orig_final) > 0:
                b0_orig_score = b0_orig_final['score'].values[0]
                b0_orig_n = int(b0_orig_final['num_scenarios'].values[0])
                match = abs(b0_our_final - b0_orig_score) < 0.001
                print(f"    batch0 original:    final_score={b0_orig_score:.6f} (n={b0_orig_n})")
                print(f"    batch0 recomputed:  final_score={b0_our_final:.6f} (n={wt})")
                print(f"    Match: {'✓ VERIFIED' if match else '✗ MISMATCH'} (diff={abs(b0_our_final - b0_orig_score):.6f})")
                
                if match:
                    print(f"    ✓ Aggregation logic matches NuPlan exactly!")
                else:
                    b0_orig_types = b0_orig_summary[b0_orig_summary['scenario_type'] != 'final_score']
                    for st, (n, s) in sorted(b0_type_scores.items()):
                        orig_row = b0_orig_types[b0_orig_types['scenario_type'] == st]
                        if len(orig_row) > 0:
                            os_val = orig_row.iloc[0]['score']
                            d = abs(s - os_val)
                            m = "✓" if d < 0.001 else "✗"
                            print(f"      {st:>50s}: orig={os_val:.6f} ours={s:.6f} {m} (diff={d:.6f})")
    
    # ===========================================================
    # Step 2c: Create merged .nuboard file
    # ===========================================================
    template_nuboards = sorted(glob.glob(str(exp_path / "latest_batch0" / "*.nuboard")))
    
    if not template_nuboards:
        print(f"  [WARNING] No .nuboard template found for {exp_name}, skipping.")
        continue
    
    template_path = template_nuboards[0]
    print(f"\n  Loading template .nuboard: {template_path}")
    
    with open(template_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    def fix_path_str(p):
        if isinstance(p, str):
            return re.sub(r'latest_batch\d+', 'merged', p)
        return p
    
    if isinstance(raw_data, dict):
        for key in raw_data:
            if isinstance(raw_data[key], str):
                old_val = raw_data[key]
                raw_data[key] = fix_path_str(raw_data[key])
                if old_val != raw_data[key]:
                    print(f"    {key}: {old_val} -> {raw_data[key]}")
    
    merged_nuboard_path = merged_path / f"{exp_name}_merged.nuboard"
    with open(merged_nuboard_path, 'wb') as f:
        pickle.dump(raw_data, f)
    
    print(f"  Saved merged .nuboard: {merged_nuboard_path}")
    nuboard_paths.append(str(merged_nuboard_path))

# Write nuboard paths
with open("/tmp/nuboard_paths.txt", "w") as f:
    f.write(",".join(nuboard_paths))

print(f"\n  All merged .nuboard paths: {nuboard_paths}")
print("  Done.")
PYEOF

# ===========================================================
# >>> 2.5  Generate comparison table & histogram
# ===========================================================
echo ""
echo "=================================================="
echo ">>> 2.5 Generating comparison report..."
echo "=================================================="

python - <<'PYEOF2'
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

base_path = Path(os.environ.get("BASE_PATH", ""))
experiments = [e.strip() for e in os.environ.get("EXP_LIST", "").split(",") if e.strip()]
challenge = os.environ.get("CHALLENGE", "closed_loop_reactive_agents")

# ===========================================================
# 1. Collect summary data from each experiment
# ===========================================================
all_summaries = {}   # exp_name -> { scenario_type: (n, score) }

for exp_name in experiments:
    agg_dir = base_path / exp_name / "merged" / "aggregator_metric"
    pq_files = sorted(agg_dir.glob("*.parquet"))
    if not pq_files:
        print(f"  [WARN] No aggregator parquet for {exp_name}")
        continue
    
    df = pd.read_parquet(pq_files[0])
    summary = df[df['num_scenarios'].notna()].copy()
    
    exp_summary = {}
    for _, row in summary.iterrows():
        stype = row['scenario_type']
        n = int(row['num_scenarios'])
        score = float(row['score']) if row['score'] is not None else 0.0
        exp_summary[stype] = (n, score)
    
    all_summaries[exp_name] = exp_summary

if not all_summaries:
    print("  No experiment data found. Skipping report.")
    exit(0)

# ===========================================================
# 2. Gather all scenario types, ensure final_score is last
# ===========================================================
all_types = set()
for exp_summary in all_summaries.values():
    all_types.update(exp_summary.keys())

all_types.discard('final_score')
sorted_types = sorted(all_types)
sorted_types.append('final_score')

# ===========================================================
# 3. Print comparison table to terminal
# ===========================================================
exp_names = list(all_summaries.keys())

print("\n" + "=" * 120)
print("  EXPERIMENT COMPARISON TABLE")
print("=" * 120)

header = f"  {'scenario_type':<55s}"
for exp_name in exp_names:
    short = exp_name[:20]
    header += f" | {'n':>5s}  {short:>20s}"
print(header)
print("  " + "-" * (len(header) - 2))

for stype in sorted_types:
    if stype == 'final_score':
        print("  " + "-" * (len(header) - 2))
    row_str = f"  {stype:<55s}"
    for exp_name in exp_names:
        if stype in all_summaries[exp_name]:
            n, score = all_summaries[exp_name][stype]
            row_str += f" | {n:>5d}  {score:>20.4f}"
        else:
            row_str += f" | {'':>5s}  {'N/A':>20s}"
    print(row_str)

print("=" * 120)

# ===========================================================
# 4. Generate comparison plots
# ===========================================================
output_dir = base_path / "comparison_plots"
output_dir.mkdir(parents=True, exist_ok=True)

colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0', '#00BCD4', '#795548', '#E91E63']

def short_label(s):
    replacements = {
        'starting_straight_traffic_light_intersection_traversal': 'straight_tl_intersect',
        'high_lateral_acceleration': 'high_lat_accel',
        'high_magnitude_speed': 'high_mag_speed',
        'low_magnitude_speed': 'low_mag_speed',
        'near_multiple_vehicles': 'near_multi_veh',
        'waiting_for_pedestrian_to_cross': 'wait_pedestrian',
        'traversing_pickup_dropoff': 'pickup_dropoff',
        'stationary_in_traffic': 'stationary',
        'following_lane_with_lead': 'follow_lead',
        'behind_long_vehicle': 'behind_long_veh',
        'starting_left_turn': 'left_turn',
        'starting_right_turn': 'right_turn',
        'stopping_with_lead': 'stop_w_lead',
        'changing_lane': 'lane_change',
    }
    return replacements.get(s, s)

types_no_final = [t for t in sorted_types if t != 'final_score']

# --- Plot A: Grouped bar chart (per-type scores) ---
fig, ax = plt.subplots(figsize=(max(16, len(types_no_final) * 1.2), 8))

x = np.arange(len(types_no_final))
bar_width = 0.8 / max(len(exp_names), 1)

for i, exp_name in enumerate(exp_names):
    scores = []
    for stype in types_no_final:
        if stype in all_summaries[exp_name]:
            scores.append(all_summaries[exp_name][stype][1])
        else:
            scores.append(0.0)
    
    offset = (i - len(exp_names) / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, scores, bar_width * 0.9,
                  label=exp_name, color=colors[i % len(colors)], alpha=0.85, edgecolor='white')
    
    for bar, score in zip(bars, scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xlabel('Scenario Type', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Per-Type Score Comparison ({challenge})', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([short_label(t) for t in types_no_final], rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 1.15)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(output_dir / "per_type_score_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir / 'per_type_score_comparison.png'}")

# --- Plot B: Final score comparison ---
fig, ax = plt.subplots(figsize=(max(6, len(exp_names) * 2), 6))

final_scores = []
final_ns = []
for exp_name in exp_names:
    if 'final_score' in all_summaries[exp_name]:
        n, score = all_summaries[exp_name]['final_score']
        final_scores.append(score)
        final_ns.append(n)
    else:
        final_scores.append(0.0)
        final_ns.append(0)

bars = ax.bar(range(len(exp_names)), final_scores,
              color=[colors[i % len(colors)] for i in range(len(exp_names))],
              alpha=0.85, edgecolor='white', width=0.6)

for bar, score, n in zip(bars, final_scores, final_ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{score:.4f}\n(n={n})', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Final Score', fontsize=12)
ax.set_title(f'Final Score Comparison ({challenge})', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(exp_names)))
ax.set_xticklabels(exp_names, fontsize=11)
ax.set_ylim(0, max(final_scores) * 1.25 if final_scores else 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(output_dir / "final_score_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir / 'final_score_comparison.png'}")

# --- Plot C: Radar chart (>= 2 experiments) ---
if len(exp_names) >= 2:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(types_no_final), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, exp_name in enumerate(exp_names):
        values = []
        for stype in types_no_final:
            if stype in all_summaries[exp_name]:
                values.append(all_summaries[exp_name][stype][1])
            else:
                values.append(0.0)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=exp_name,
                color=colors[i % len(colors)], markersize=5)
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([short_label(t) for t in types_no_final], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f'Radar: Per-Type Score ({challenge})', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'radar_comparison.png'}")

# --- Plot D: Heatmap ---
fig, ax = plt.subplots(figsize=(max(14, len(types_no_final) * 0.9), max(3, len(exp_names) * 1.5 + 2)))

heatmap_data = []
for exp_name in exp_names:
    row = []
    for stype in types_no_final:
        if stype in all_summaries[exp_name]:
            row.append(all_summaries[exp_name][stype][1])
        else:
            row.append(np.nan)
    heatmap_data.append(row)

heatmap_array = np.array(heatmap_data)
im = ax.imshow(heatmap_array, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(types_no_final)))
ax.set_xticklabels([short_label(t) for t in types_no_final], rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(exp_names)))
ax.set_yticklabels(exp_names, fontsize=11)

for i in range(len(exp_names)):
    for j in range(len(types_no_final)):
        val = heatmap_array[i, j]
        if not np.isnan(val):
            text_color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')

ax.set_title(f'Score Heatmap ({challenge})', fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8, label='Score')
plt.tight_layout()
fig.savefig(output_dir / "score_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {output_dir / 'score_heatmap.png'}")

print(f"\n  All plots saved to: {output_dir}")
PYEOF2

# Read back the nuboard paths
NUBOARD_PATHS=$(cat /tmp/nuboard_paths.txt)
IFS=',' read -ra PATH_ARRAY <<< "$NUBOARD_PATHS"

# Build hydra simulation_path
FILE_LIST=""
for p in "${PATH_ARRAY[@]}"; do
    FILE_LIST="$FILE_LIST'$p',"
done
FILE_LIST=${FILE_LIST%,}
HYDRA_SIM_PATH="[$FILE_LIST]"

echo ""
echo "=================================================="
echo ">>> 3. Starting NuBoard (comparing ${#EXPERIMENTS[@]} experiments)..."
echo ">>> Experiments: ${EXPERIMENTS[*]}"
echo ">>> .nuboard files: $HYDRA_SIM_PATH"
echo "=================================================="

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_nuboard.py \
    simulation_path="$HYDRA_SIM_PATH" \
    scenario_builder=nuplan \
    +port=5006