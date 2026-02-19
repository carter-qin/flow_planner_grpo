import re
import os
import sys
from typing import Literal, Callable, Any, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from flow_planner.model.model_base import DiffusionADPlanner
from flow_planner.model.model_utils.input_preprocess import ModelInputProcessor
from flow_planner.model.model_utils.traj_tool import traj_chunking, assemble_actions
from flow_planner.data.dataset.nuplan import NuPlanDataSample


class FlowPlanner(DiffusionADPlanner):
    def __init__(
        self,
        model_encoder,
        model_decoder,
        flow_ode,
        flow_sde,  # flow grpo sde with logprob
        model_type: Literal["x_start", "noise", "velocity"] = "x_start",
        kinematic: Literal["waypoints", "velocity", "acceleration"] = "waypoints",
        assemble_method="linear",
        data_processor: ModelInputProcessor = None,
        device="cuda",
        **planner_params,
    ):

        super(FlowPlanner, self).__init__()
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self._model_type = model_type
        self.device = device

        self.flow_ode = flow_ode  # including flow matching path and ode solver
        self.flow_sde = flow_sde  # flow grpo sde with logprob
        self.cfg_prob = planner_params["cfg_prob"]
        self.cfg_weight = planner_params["cfg_weight"]
        self.cfg_type = planner_params["cfg_type"]

        self.kinematic = kinematic

        self.assemble_method = assemble_method

        self.data_processor = data_processor

        self.planner_params = (
            planner_params  # including the action_len, future_len etc.
        )
        self.action_num = (
            self.planner_params["future_len"] - self.planner_params["action_overlap"]
        ) // (self.planner_params["action_len"] - self.planner_params["action_overlap"])

        self.basic_loss = nn.MSELoss(reduction="none")

    # ===================================================
    # Shared utilities
    # ===================================================

    @staticmethod
    def _expand_data(data, group_size):
        """Recursively expand tensor data by repeating along batch dim."""
        if isinstance(data, torch.Tensor):
            return data.repeat_interleave(group_size, dim=0)
        elif isinstance(data, (tuple, list)):
            return type(data)(
                [FlowPlanner._expand_data(item, group_size) for item in data]
            )
        elif isinstance(data, dict):
            return {k: FlowPlanner._expand_data(v, group_size) for k, v in data.items()}
        return data

    def prepare_decoder_extra(self, data: NuPlanDataSample, group_size=1):
        """
        Prepare decoder_model_extra from data. Optionally expand for group sampling.
        Used by both forward_rollout and trainer_rl to avoid duplicating
        encoder + expand logic.

        Args:
            data: input data sample
            group_size: number of group samples per batch item

        Returns:
            decoder_model_extra: dict ready to pass to self.decoder()
        """
        B = data.ego_current.shape[0]
        cfg_flags = torch.ones((B,), device=self.device).to(torch.int32)

        model_inputs, _ = self.prepare_model_input(
            cfg_flags, data, use_cfg=False, is_training=False
        )

        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        with torch.no_grad():
            encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)

        if group_size > 1:
            decoder_model_extra = self._expand_data(decoder_model_extra, group_size)

        return decoder_model_extra

    # ===================================================
    # Input preparation
    # ===================================================

    def prepare_model_input(
        self, cfg_flags, data: NuPlanDataSample, use_cfg, is_training
    ):
        B = data.ego_current.shape[0]

        if is_training:
            # modify the data sample according to cfg_flags
            cfg_type = self.cfg_type
            if cfg_type == "neighbors":
                neighbor_num = self.planner_params["neighbor_num"]
                cfg_neighbor_num = min(
                    self.planner_params["cfg_neighbor_num"], neighbor_num
                )
                mask_flags = cfg_flags.view(
                    B, *([1] * (data.neighbor_past.dim() - 1))
                ).repeat(1, neighbor_num, 1, 1)
                mask_flags[:, cfg_neighbor_num:, :] = 1
                data.neighbor_past *= mask_flags
            elif cfg_type == "lanes":
                data.lanes = data.lanes * cfg_flags.view(
                    B, *([1] * (data.lanes.dim() - 1))
                )

        else:
            if use_cfg:
                data = data.repeat(2)
                cfg_type = self.cfg_type
                if cfg_type == "neighbors":
                    neighbor_num = self.planner_params["neighbor_num"]
                    cfg_neighbor_num = min(
                        self.planner_params["cfg_neighbor_num"], neighbor_num
                    )
                    mask_flags = cfg_flags.view(
                        B * 2, *([1] * (data.neighbor_past.dim() - 1))
                    ).repeat(1, neighbor_num, 1, 1)
                    mask_flags[:, cfg_neighbor_num:, :] = 1
                    data.neighbor_past *= mask_flags
                elif cfg_type == "lanes":
                    data.lanes = data.lanes * cfg_flags.view(
                        B * 2, *([1] * (data.lanes.dim() - 1))
                    )

        model_inputs, gt = self.data_processor.sample_to_model_input(
            data, device=self.device, kinematic=self.kinematic, is_training=is_training
        )

        model_inputs.update({"cfg_flags": cfg_flags})

        return model_inputs, gt

    def extract_encoder_inputs(self, inputs):

        encoder_inputs = {
            "neighbors": inputs["neighbor_past"],
            "lanes": inputs["lanes"],
            "lanes_speed_limit": inputs["lanes_speedlimit"],
            "lanes_has_speed_limit": inputs["lanes_has_speedlimit"],
            "static": inputs["map_objects"],
            "routes": inputs["routes"],
        }
        return encoder_inputs

    def extract_decoder_inputs(self, encoder_outputs, inputs):
        model_extra = dict(
            cfg_flags=inputs["cfg_flags"] if "cfg_flags" in inputs.keys() else None,
        )
        model_extra.update(encoder_outputs)
        return model_extra

    def encoder(self, **encoder_inputs):
        return self.model_encoder(**encoder_inputs)

    def decoder(self, x, t, **model_extra):
        return self.model_decoder(x, t, **model_extra)

    # ===================================================
    # Forward dispatch
    # ===================================================

    def forward(self, data: NuPlanDataSample, mode="train", **params):
        if mode == "train":
            return self.forward_train(data)
        elif mode == "inference":
            return self.forward_inference(data, params["use_cfg"], params["cfg_weight"])
        elif mode == "rollout":
            return self.forward_rollout(
                data,
                group_size=params.get("group_size", 8),
            )

    # ===================================================
    # Training forward
    # ===================================================

    def forward_train(self, data: NuPlanDataSample):
        """
        Forward a training step and compute the training loss.
        1. generate cfg_flags
        2. preprocess (masking) according to the cfg_flags
        3. model forward
        4. compute basic mse loss

        Return:
            prediction: the raw prediction of the model, specified by model.prediction_type;
            loss_dict: a dict of loss containing unreduced mse loss, consistency loss and neighbor prediction loss (if one exists).
        """
        B = data.ego_current.shape[0]
        roll_dice = torch.rand((B, 1))
        cfg_flags = (
            (roll_dice > self.cfg_prob).to(torch.int32).to(self.device)
        )  # NOTE: 1 for conditioned (unmasked), 0 for unconditioned (masked)
        model_inputs, gt = self.prepare_model_input(
            cfg_flags, data, use_cfg=False, is_training=True
        )  # note that the cfg_flags are packed into the model_inputs

        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        B, P, T_, D = gt.shape

        noised_traj, target, t = self.flow_ode.sample(gt[:, :, 1:, :], self._model_type)
        noised_traj_tokens = traj_chunking(
            noised_traj,
            self.planner_params["action_len"],
            self.planner_params["action_overlap"],
        )
        noised_traj_tokens = torch.cat(noised_traj_tokens, dim=1)
        target_tokens = traj_chunking(
            target,
            self.planner_params["action_len"],
            self.planner_params["action_overlap"],
        )
        target_tokens = torch.cat(target_tokens, dim=1)

        prediction = self.decoder(noised_traj_tokens, t, **decoder_model_extra)

        loss_dict = {}
        batch_loss = self.basic_loss(prediction, target_tokens)
        loss_dict["batch_loss"] = batch_loss

        loss = torch.sum(batch_loss, dim=-1)  # (B, action_num, action_length, dim)
        loss_dict["ego_planning_loss"] = loss.mean()

        if self.planner_params["action_overlap"] > 0:
            consistency_loss = [
                torch.mean(
                    torch.sum(
                        self.basic_loss(
                            prediction[
                                :,
                                i : i + 1,
                                -self.planner_params["action_overlap"] :,
                                :,
                            ],
                            prediction[
                                :,
                                i + 1 : i + 2,
                                : self.planner_params["action_overlap"],
                                :,
                            ],
                        ),
                        dim=-1,
                    )
                )
                for i in range(0, prediction.shape[1] - 2)
            ]
            loss_dict["consistency_loss"] = sum(consistency_loss) / len(
                consistency_loss
            )
        else:
            loss_dict["consistency_loss"] = torch.tensor(0.0, device=loss.device)

        assert not torch.isnan(loss).sum(), f"loss is NaN"

        return prediction, loss_dict

    # ===================================================
    # RL Rollout forward
    # ===================================================

    def forward_rollout(self, data: NuPlanDataSample, group_size=8):
        """
        [RL Training] Rollout with SDE to generate samples + log probabilities.

        Returns:
            sample_assembled: (B*G, future_len, state_dim)
            all_samples: (B*G, steps+1, action_num, action_len, state_dim)
            all_log_probs: (B*G, steps)
            all_prev_sample_means: (B*G, steps, action_num, action_len, state_dim)
            all_std_dev_t: (steps,) scalar per step
            decoder_model_extra: dict, reusable in Phase 2 (avoids redundant encoder forward)
        """
        B = data.ego_current.shape[0]

        decoder_model_extra_expanded = self.prepare_decoder_extra(
            data, group_size=group_size
        )

        x_init = torch.randn(
            (
                B * group_size,
                self.action_num,
                self.planner_params["action_len"],
                self.planner_params["state_dim"],
            ),
            device=self.device,
        )
        all_samples, all_log_probs, all_prev_sample_means, all_std_dev_t = (
            self.flow_sde.generate(
                x_init,
                self.decoder,
                self._model_type,
                **decoder_model_extra_expanded,
            )
        )

        sample_assembled = assemble_actions(
            all_samples[:, -1],
            self.planner_params["future_len"],
            self.planner_params["action_len"],
            self.planner_params["action_overlap"],
            self.planner_params["state_dim"],
            self.assemble_method,
        )

        sample_assembled = self.data_processor.state_postprocess(sample_assembled)

        if sample_assembled is not None and sample_assembled.dim() == 4:
            B, G, T, D = sample_assembled.shape
            sample_assembled = sample_assembled.reshape(B * G, T, D)

        return (
            sample_assembled,
            all_samples,
            all_log_probs,
            all_prev_sample_means,
            all_std_dev_t,
            decoder_model_extra_expanded,
        )

    # ===================================================
    # Inference forward (keeps use_cfg for eval)
    # ===================================================

    def forward_inference(self, data: NuPlanDataSample, use_cfg=True, cfg_weight=None):
        B = data.ego_current.shape[0]
        if use_cfg:
            cfg_flags = torch.cat(
                [
                    torch.ones((B,), device=self.device),
                    torch.zeros((B,), device=self.device),
                ],
                dim=0,
            ).to(torch.int32)
        else:
            cfg_flags = torch.ones((B,), device=self.device).to(torch.int32)

        model_inputs, _ = self.prepare_model_input(
            cfg_flags, data, use_cfg, is_training=False
        )

        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)

        x_init = torch.randn(
            (
                B,
                self.action_num,
                self.planner_params["action_len"],
                self.planner_params["state_dim"],
            ),
            device=self.device,
        )
        sample = self.flow_ode.generate(
            x_init,
            self.decoder,
            self._model_type,
            use_cfg=use_cfg,
            cfg_weight=cfg_weight,
            **decoder_model_extra,
        )

        sample = assemble_actions(
            sample,
            self.planner_params["future_len"],
            self.planner_params["action_len"],
            self.planner_params["action_overlap"],
            self.planner_params["state_dim"],
            self.assemble_method,
        )

        sample = self.data_processor.state_postprocess(sample)

        return sample

    # ===================================================
    # Properties & helpers
    # ===================================================

    @property
    def model_type(
        self,
    ):
        return self._model_type

    def get_optimizer_params(self):
        """Returns optimizer param groups. For LoRA training, only trainable params are included."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if trainable_params:
            return [{"params": trainable_params}]
        # Fallback: all params (SFT mode)
        return [
            {"params": self.model_encoder.parameters()},
            {"params": self.model_decoder.parameters()},
        ]
