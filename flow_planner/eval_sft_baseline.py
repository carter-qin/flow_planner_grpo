"""
Quick script to evaluate SFT baseline (LoRA weights = 0) on same fixed scenes as RL eval.
Run: python -m flow_planner.eval_sft_baseline
"""

import torch
import copy
from flow_planner.model.rewards import RewardConfig, compute_rewards_and_advantages


def eval_sft_baseline(model, eval_data_list, device, group_size=4):
    """Evaluate model with LoRA weights zeroed out (= pure SFT)."""
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Zero out all LoRA weights
    for name, p in model_copy.named_parameters():
        if "lora" in name.lower():
            p.data.zero_()

    reward_config = RewardConfig(dt=0.1)
    eval_rewards = []

    with torch.no_grad():
        for eval_data in eval_data_list:
            eval_data = eval_data.to(device)
            (samples, _, _, _, _, _) = model_copy(
                eval_data,
                mode="rollout",
                group_size=group_size,
            )
            gt_3d = eval_data.ego_future[:, :, :3]
            gt_x, gt_y, gt_h = gt_3d[..., 0:1], gt_3d[..., 1:2], gt_3d[..., 2:3]
            gt_4d = torch.cat([gt_x, gt_y, torch.cos(gt_h), torch.sin(gt_h)], dim=-1)
            gt_exp = gt_4d.repeat_interleave(group_size, dim=0)

            nf = None
            if (
                hasattr(eval_data, "neighbor_future_observed")
                and eval_data.neighbor_future_observed is not None
            ):
                nf = eval_data.neighbor_future_observed.repeat_interleave(
                    group_size, dim=0
                )

            r, _, _ = compute_rewards_and_advantages(
                samples, gt_exp, nf, reward_config, group_size
            )
            eval_rewards.append(r.mean().item())

    return sum(eval_rewards) / len(eval_rewards)
