import os
import argparse
import json
import math
from multiprocessing import Pool

from flow_planner.data.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)


def get_filter_parameters(
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    shuffle=True,
    scenario_tokens=None,
    log_names=None,
):
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def process_data_chunk(scenarios_chunk, args):
    # --- Resume (断点续传) 逻辑 ---
    scenarios_to_process = []
    for scenario in scenarios_chunk:
        map_name = scenario._map_name
        token = scenario.token
        # 检查文件是否已存在
        file_path = os.path.join(args.save_path, f"{map_name}_{token}.npz")
        if not os.path.exists(file_path):
            scenarios_to_process.append(scenario)

    if len(scenarios_to_process) == 0:
        return

    # [核心修正] 这里只传 save_path 字符串
    processor = DataProcessor(args.save_path)
    processor.work(scenarios_to_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument("--data_path", default="/data/nuplan-v1.1/trainval", type=str)
    parser.add_argument("--map_path", default="/data/nuplan-v1.1/maps", type=str)
    parser.add_argument("--save_path", default="./cache", type=str)
    parser.add_argument("--scenarios_per_type", type=int, default=None)
    parser.add_argument("--total_scenarios", type=int, default=10)
    parser.add_argument(
        "--shuffle_scenarios", type=int, default=1
    )  # argparse bool trick

    # 这些参数虽然 DataProcessor 没用到 (它硬编码了)，但保留着不影响
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_num", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=32)

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    sensor_root = None
    db_files = None

    # [可选] 如果你目录下没有 nuplan_train.json，这行会报错
    # 如果想跑全量，可以将 log_names 设为 None
    log_names = None
    if "train" in args.data_path and os.path.exists("./nuplan_train.json"):
        with open("./nuplan_train.json", "r", encoding="utf-8") as file:
            log_names = json.load(file)

    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(
        args.data_path, args.map_path, sensor_root, db_files, map_version
    )

    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            args.scenarios_per_type,
            args.total_scenarios,
            bool(args.shuffle_scenarios),
            log_names=log_names,
        )
    )

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # [新增] 核心加速：每 5 帧取 1 帧
    # 100万 -> 20万，处理速度快 5 倍，训练速度快 5 倍
    scenarios = scenarios[::1]
    print(f"Downsampled scenarios (1/1): {len(scenarios)}")

    # process data
    del worker, builder, scenario_filter

    num_proc = args.num_workers
    print(f"Starting processing with {num_proc} workers...")

    if len(scenarios) > 0:
        chunk_size = math.ceil(len(scenarios) / num_proc)
        scenarios_chunks = [
            scenarios[i : i + chunk_size] for i in range(0, len(scenarios), chunk_size)
        ]
        pool_args = [(chunk, args) for chunk in scenarios_chunks]

        with Pool(processes=num_proc) as pool:
            pool.starmap(process_data_chunk, pool_args)
    else:
        print("No scenarios found to process.")

    # [优化] 根据 save_path 自动命名 json 文件，防止覆盖
    split_name = os.path.basename(os.path.normpath(args.save_path))
    output_json = f"{split_name}_files.json"

    npz_files = [f for f in os.listdir(args.save_path) if f.endswith(".npz")]
    with open(output_json, "w") as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names to {output_json}")
