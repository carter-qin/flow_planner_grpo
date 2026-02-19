
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type
import hydra
from hydra.utils import instantiate
import omegaconf

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from flow_planner.data.data_process.data_processor import DataProcessor
from flow_planner.data.dataset.nuplan import NuPlanDataSample

def identity(ego_state, predictions):
    return predictions


class FlowPlanner(AbstractPlanner):
    def __init__(
            self,
            config_path,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            enable_lora: bool = False,
            device: str = "cpu",
            use_cfg: bool = True,
            cfg_weight: float = 1.0,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        config = omegaconf.OmegaConf.load(config_path)
        self._config = config

        self._enable_lora = enable_lora

        # Strip LoRA configs only when NOT using LoRA.
        # When enable_lora=True, we keep the LoRA layers so that
        # LoRA checkpoint weights can be loaded into them.
        if not enable_lora:
            with omegaconf.open_dict(config):
                if "model" in config:
                    # decoder-level lora_configs
                    if "model_decoder" in config.model and "lora_configs" in config.model.model_decoder:
                        config.model.model_decoder.lora_configs = None
                    # Also handle any nested lora_config keys
                    def _strip_lora(cfg):
                        if isinstance(cfg, omegaconf.DictConfig):
                            for key in list(cfg.keys()):
                                if key in ("lora_config", "lora_configs"):
                                    cfg[key] = None
                                else:
                                    _strip_lora(cfg[key])
                        elif isinstance(cfg, omegaconf.ListConfig):
                            for item in cfg:
                                _strip_lora(item)
                    _strip_lora(config.model)

        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = instantiate(config.model)

        self.core = instantiate(config.core)

        self.data_processor = DataProcessor(None)

        self.use_cfg = use_cfg

        self.cfg_weight = cfg_weight
        
    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            ckpt = torch.load(self._ckpt_path, map_location="cpu", weights_only=False)

            # Determine which state_dict to use:
            # 1. EMA weights (preferred for inference)
            # 2. Model weights as fallback
            if self._ema_enabled and "ema_state_dict" in ckpt:
                raw_sd = ckpt["ema_state_dict"]
            elif "model" in ckpt:
                raw_sd = ckpt["model"]
            else:
                # Legacy format: ckpt itself is the state_dict
                raw_sd = ckpt

            # Strip DDP "module." prefix if present
            model_state_dict = {
                k.replace("module.", "", 1): v for k, v in raw_sd.items()
            }

            if self._enable_lora:
                # LoRA mode: load base weights first with strict=False (LoRA params will be missing),
                # then overlay LoRA weights from the checkpoint.
                
                # Separate base weights and LoRA weights from checkpoint
                lora_weights = {k: v for k, v in model_state_dict.items() if "lora" in k.lower()}
                base_weights = {k: v for k, v in model_state_dict.items() if "lora" not in k.lower()}
                
                # Load base weights (LoRA params in model will keep their init values)
                missing_base, unexpected_base = self._planner.load_state_dict(base_weights, strict=False)
                
                # Filter out expected LoRA missing keys
                non_lora_missing = [k for k in missing_base if "lora" not in k.lower()]
                if non_lora_missing:
                    print(f"[WARNING] Non-LoRA missing keys ({len(non_lora_missing)}): {non_lora_missing[:10]}")
                if unexpected_base:
                    print(f"[WARNING] Unexpected keys from base weights ({len(unexpected_base)}): {unexpected_base[:10]}")
                
                # Now load LoRA weights if present in checkpoint
                if lora_weights:
                    lora_missing, lora_unexpected = self._planner.load_state_dict(
                        lora_weights, strict=False
                    )
                    # lora_missing will contain ALL non-lora keys (expected)
                    actual_lora_missing = [k for k in lora_missing if "lora" in k.lower()]
                    if actual_lora_missing:
                        print(f"[WARNING] LoRA keys in model but NOT in checkpoint ({len(actual_lora_missing)}): {actual_lora_missing[:10]}")
                    print(f"LoRA checkpoint loaded. LoRA weights loaded: {len(lora_weights)} | LoRA missing in ckpt: {len(actual_lora_missing)}")
                else:
                    print("[INFO] No LoRA weights found in checkpoint. LoRA layers use random initialization (likely an SFT checkpoint).")
                
                # Also try loading LoRA-only state dict if saved separately
                if "lora_state_dict" in ckpt and ckpt["lora_state_dict"]:
                    lora_only_sd = ckpt["lora_state_dict"]
                    lora_only_sd = {k.replace("module.", "", 1): v for k, v in lora_only_sd.items()}
                    lm, lu = self._planner.load_state_dict(lora_only_sd, strict=False)
                    actual_lora_loaded = len(lora_only_sd) - len([k for k in lu if "lora" in k.lower()])
                    print(f"[INFO] Also loaded lora_state_dict key: {len(lora_only_sd)} entries")
            else:
                # Non-LoRA mode: original loading logic
                missing, unexpected = self._planner.load_state_dict(model_state_dict, strict=False)

                non_lora_missing = [k for k in missing if "lora" not in k.lower()]
                if non_lora_missing:
                    print(f"[WARNING] Non-LoRA missing keys ({len(non_lora_missing)}): {non_lora_missing[:10]}")
                if unexpected:
                    print(f"[WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:10]}")

                lora_missing = [k for k in missing if "lora" in k.lower()]
                print(
                    f"Checkpoint loaded. "
                    f"LoRA missing (expected if merged): {len(lora_missing)} | "
                    f"Non-LoRA missing: {len(non_lora_missing)} | "
                    f"Unexpected: {len(unexpected)}"
                )
        else:
            print("[WARNING] No checkpoint path provided, using random weights.")

        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        data = NuPlanDataSample(
            batched=(model_inputs['ego_current_state'].dim() > 1),
            ego_past=model_inputs['ego_agent_past'],
            ego_current=model_inputs['ego_current_state'],
            neighbor_past=model_inputs['neighbor_agents_past'],
            lanes=model_inputs['lanes'],
            lanes_speedlimit=model_inputs['lanes_speed_limit'],
            lanes_has_speedlimit=model_inputs['lanes_has_speed_limit'],
            routes=model_inputs['route_lanes'],
            routes_speedlimit=model_inputs['route_lanes_speed_limit'],
            routes_has_speedlimit=model_inputs['route_lanes_has_speed_limit'],
            map_objects=model_inputs['static_objects']
        )

        return data

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    
        predictions = outputs[0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        outputs = self.core.inference(self._planner, inputs, use_cfg=self.use_cfg, cfg_weight=self.cfg_weight)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
    