"""
Reward functions for Flow-GRPO RL training.

Trajectory format: (B, T, 4) where 4 = (x, y, cos_theta, sin_theta)
All reward functions return (B,) tensors, higher is better.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class RewardConfig:
    """Weights and thresholds for each reward component."""

    w_collision: float = 0.0  # 不用加法项
    collision_threshold: float = 2.3
    collision_type: str = "soft"

    w_progress: float = 1.0  # 恢复 1.0
    position_norm: str = "ade"

    w_heading: float = 0.0  # 关闭

    w_smoothness: float = 0.0
    dt: float = 0.1
    acc_threshold: float = 6.0

    w_comfort: float = 0.0
    jerk_threshold: float = 20.0
    lat_acc_threshold: float = 4.0

    reward_clip: float = 15.0


# ============================================================
# Individual reward components
# ============================================================


def reward_collision(
    pred_traj: Tensor,
    neighbor_future: Tensor,
    threshold: float = 2.3,
    collision_type: str = "soft",
    dt: float = 0.1,
) -> Tensor:
    """
    基于双向多圆模型 + 速度启发式的碰撞检测。
    """
    T = pred_traj.shape[1]
    device = pred_traj.device

    ego_xy = pred_traj[:, :, :2]
    ego_cos = pred_traj[:, :, 2:3]
    ego_sin = pred_traj[:, :, 3:4]

    ego_offsets = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0], device=device).view(
        1, 1, 5, 1
    )

    ego_cx = ego_xy[..., 0:1].unsqueeze(-2) + ego_offsets * ego_cos.unsqueeze(-2)
    ego_cy = ego_xy[..., 1:2].unsqueeze(-2) + ego_offsets * ego_sin.unsqueeze(-2)
    ego_circles = torch.cat([ego_cx, ego_cy], dim=-1)

    neigh_xy = neighbor_future[:, :, :T, :2]
    neigh_mask = (neigh_xy != 0).any(dim=-1)

    neigh_heading = neighbor_future[:, :, :T, 2]
    neigh_cos = torch.cos(neigh_heading).unsqueeze(-1)
    neigh_sin = torch.sin(neigh_heading).unsqueeze(-1)

    diffs = neigh_xy[:, :, 1:, :] - neigh_xy[:, :, :-1, :]
    speed = torch.norm(diffs, dim=-1) / dt

    if speed.shape[2] > 0:
        speed = torch.cat([speed, speed[:, :, -1:]], dim=2)
    else:
        speed = torch.zeros_like(neigh_mask, dtype=torch.float32)

    scale = torch.clamp((speed - 1.0) / 3.0, min=0.0, max=1.0)
    scale = scale.unsqueeze(-1).unsqueeze(-1)

    base_offsets = torch.tensor([1.5, 0.0, -1.5], device=device).view(1, 1, 1, 3, 1)
    neigh_offsets = base_offsets * scale

    neigh_cx = neigh_xy[..., 0:1].unsqueeze(-2) + neigh_offsets * neigh_cos.unsqueeze(
        -2
    )
    neigh_cy = neigh_xy[..., 1:2].unsqueeze(-2) + neigh_offsets * neigh_sin.unsqueeze(
        -2
    )
    neigh_circles = torch.cat([neigh_cx, neigh_cy], dim=-1)

    ego_expanded = ego_circles.unsqueeze(1).unsqueeze(-2)
    neigh_expanded = neigh_circles.unsqueeze(-3)

    diff = ego_expanded - neigh_expanded
    dist = torch.norm(diff, dim=-1)

    min_dist = dist.min(dim=-1)[0].min(dim=-1)[0]

    min_dist = min_dist.masked_fill(~neigh_mask, float("inf"))

    if collision_type == "hard":
        is_collision = (min_dist < threshold).any(dim=-1).any(dim=-1).float()
        reward = -is_collision
    else:
        penetration = torch.clamp(threshold - min_dist, min=0)
        penalty = (penetration**2).sum(dim=-1).sum(dim=-1) / T
        reward = -penalty

    return reward


def reward_progress_and_tracking(
    pred_traj: Tensor,
    gt_traj: Tensor,
    dt: float = 0.1,
) -> Tensor:
    """
    Progress + ADE + FDE reward — 恢复宽分母，保证组内有梯度。
    """
    pred_xy = pred_traj[:, :, :2]
    gt_xy = gt_traj[:, :, :2]
    T = pred_xy.shape[1]

    # --- 1. Progress ---
    gt_vec = gt_xy[:, -1, :] - gt_xy[:, 0, :]
    gt_dist = torch.norm(gt_vec, dim=-1, keepdim=True) + 1e-5
    gt_dir = gt_vec / gt_dist

    pred_vec = pred_xy[:, -1, :] - pred_xy[:, 0, :]
    pred_progress = (pred_vec * gt_dir).sum(dim=-1)
    gt_progress = gt_dist.squeeze(-1)
    progress_error = torch.abs(pred_progress - gt_progress)

    # --- 2. ADE ---
    displacement = torch.norm(pred_xy - gt_xy, dim=-1)
    weights = torch.linspace(0.5, 1.5, T, device=pred_xy.device).unsqueeze(0)
    ade = (displacement * weights).mean(dim=-1)

    # --- 3. FDE ---
    fde = torch.norm(pred_xy[:, -1, :] - gt_xy[:, -1, :], dim=-1)

    # [修改] 用纯高斯核，没有 clamp=0 的死区
    # sigma 选择让 error=0 时 reward=1，error=4m 时 reward≈0.37
    r_progress = torch.exp(-(progress_error**2) / (2 * 3.0**2))
    r_ade = torch.exp(-(ade**2) / (2 * 3.0**2))
    r_fde = torch.exp(-(fde**2) / (2 * 4.0**2))

    reward = 0.2 * r_progress + 0.45 * r_ade + 0.35 * r_fde

    return reward


def reward_heading_tracking(
    pred_traj: Tensor,
    gt_traj: Tensor,
) -> Tensor:
    """
    Heading tracking reward: 正向高斯核，返回 [0, 1]。
    """
    pred_cos, pred_sin = pred_traj[:, :, 2], pred_traj[:, :, 3]
    gt_cos, gt_sin = gt_traj[:, :, 2], gt_traj[:, :, 3]

    pred_norm = torch.hypot(pred_cos, pred_sin) + 1e-8
    pred_cos = pred_cos / pred_norm
    pred_sin = pred_sin / pred_norm

    cos_delta = pred_cos * gt_cos + pred_sin * gt_sin
    cos_delta = torch.clamp(cos_delta, -1.0, 1.0)

    angle_error = torch.acos(cos_delta)
    mean_error = angle_error.mean(dim=-1)

    reward = torch.exp(-(mean_error**2) / (2 * 0.3**2))

    return reward


def reward_smoothness(
    pred_traj: Tensor,
    dt: float = 0.1,
    acc_threshold: float = 2.0,
    mask: Tensor | None = None,
) -> Tensor:
    """
    使用多步差分 (Macro-Kinematics) 消除生成模型高阶导数灾难。
    返回负值 (惩罚项)。
    """
    k = 4
    B, T, _ = pred_traj.shape
    if T <= 2 * k:
        return torch.zeros(B, device=pred_traj.device)

    xy = pred_traj[:, :, :2]
    dt_k = k * dt

    xy_permuted = xy.permute(0, 2, 1)
    xy_padded = F.pad(xy_permuted, (1, 1), mode="replicate")
    xy_smoothed = F.avg_pool1d(xy_padded, kernel_size=3, stride=1).permute(0, 2, 1)

    vel = (xy_smoothed[:, k:] - xy_smoothed[:, :-k]) / dt_k
    acc = (vel[:, k:] - vel[:, :-k]) / dt_k

    acc_mag = torch.norm(acc, dim=-1)
    acc_excess = torch.clamp(acc_mag - acc_threshold, min=0)

    if mask is not None:
        valid_mask = mask[:, 2 * k :]
        penalty = (acc_excess * valid_mask).sum(dim=-1) / (
            valid_mask.sum(dim=-1) + 1e-5
        )
    else:
        penalty = acc_excess.mean(dim=-1)

    return -penalty


def reward_comfort(
    pred_traj: Tensor,
    dt: float = 0.1,
    jerk_threshold: float = 1.5,
    lat_acc_threshold: float = 1.5,
    mask: Tensor | None = None,
) -> Tensor:
    """
    舒适度惩罚：结合多步差分与平滑，计算宏观 Jerk 与横向加速度。
    返回负值 (惩罚项)。
    """
    k = 4
    B, T, _ = pred_traj.shape
    if T <= 3 * k:
        return torch.zeros(B, device=pred_traj.device)

    xy = pred_traj[:, :, :2]
    cos_theta = pred_traj[:, :, 2]
    sin_theta = pred_traj[:, :, 3]
    dt_k = k * dt

    xy_permuted = xy.permute(0, 2, 1)
    xy_padded = F.pad(xy_permuted, (1, 1), mode="replicate")
    xy_smoothed = F.avg_pool1d(xy_padded, kernel_size=3, stride=1).permute(0, 2, 1)

    vel = (xy_smoothed[:, k:] - xy_smoothed[:, :-k]) / dt_k
    acc = (vel[:, k:] - vel[:, :-k]) / dt_k
    jerk = (acc[:, k:] - acc[:, :-k]) / dt_k

    jerk_mag = torch.norm(jerk, dim=-1)
    jerk_excess = torch.clamp(jerk_mag - jerk_threshold, min=0)

    sin_dtheta = (
        cos_theta[:, :-k] * sin_theta[:, k:] - sin_theta[:, :-k] * cos_theta[:, k:]
    )
    cos_dtheta = (
        cos_theta[:, :-k] * cos_theta[:, k:] + sin_theta[:, :-k] * sin_theta[:, k:]
    )
    dtheta = torch.atan2(sin_dtheta, cos_dtheta)

    yaw_rate = dtheta / dt_k
    speed = torch.norm(vel, dim=-1)
    lat_acc = torch.abs(speed * yaw_rate)

    lat_excess = torch.clamp(lat_acc - lat_acc_threshold, min=0)

    if mask is not None:
        jerk_mask = mask[:, 3 * k :]
        lat_mask = mask[:, k:]
        jerk_penalty = (jerk_excess * jerk_mask).sum(dim=-1) / (
            jerk_mask.sum(dim=-1) + 1e-5
        )
        lat_penalty = (lat_excess * lat_mask).sum(dim=-1) / (
            lat_mask.sum(dim=-1) + 1e-5
        )
    else:
        jerk_penalty = jerk_excess.mean(dim=-1)
        lat_penalty = lat_excess.mean(dim=-1)

    return -(jerk_penalty + lat_penalty)


# ============================================================
# Combined reward function
# ============================================================


def compute_reward(
    pred_traj: Tensor,
    gt_traj: Tensor,
    neighbor_future: Tensor | None = None,
    config: RewardConfig | None = None,
    group_size: int = 1,
):
    """
    组合奖励函数 — 纯 progress reward。
    """
    if config is None:
        config = RewardConfig()

    B = pred_traj.shape[0]
    device = pred_traj.device

    T_pred = pred_traj.shape[1]
    T_gt = gt_traj.shape[1]
    T = min(T_pred, T_gt)
    pred_traj = pred_traj[:, :T]
    gt_traj = gt_traj[:, :T]

    ego_mask = (gt_traj[:, :, :2] != 0).any(dim=-1)

    components = {}

    # 1. Collision — 仅日志
    r_collision = torch.zeros(B, device=device)
    if neighbor_future is not None:
        r_collision = reward_collision(
            pred_traj,
            neighbor_future,
            threshold=config.collision_threshold,
            collision_type=config.collision_type,
            dt=config.dt,
        )
        components["collision"] = r_collision

    # 2. Progress
    r_progress = torch.zeros(B, device=device)
    if config.w_progress > 0:
        r_progress = reward_progress_and_tracking(
            pred_traj,
            gt_traj,
            dt=config.dt,
        )
        components["progress"] = r_progress

    # 3. Heading — 仅日志
    r_heading = torch.zeros(B, device=device)
    r_heading = reward_heading_tracking(pred_traj, gt_traj)
    components["heading"] = r_heading

    # 4. Smoothness
    r_smooth = torch.zeros(B, device=device)
    if config.w_smoothness > 0 and T >= 3:
        r_smooth = reward_smoothness(
            pred_traj, dt=config.dt, acc_threshold=config.acc_threshold, mask=ego_mask
        )
        components["smoothness"] = r_smooth

    # 5. Comfort
    r_comfort = torch.zeros(B, device=device)
    if config.w_comfort > 0 and T >= 4:
        r_comfort = reward_comfort(
            pred_traj,
            dt=config.dt,
            jerk_threshold=config.jerk_threshold,
            lat_acc_threshold=config.lat_acc_threshold,
            mask=ego_mask,
        )
        components["comfort"] = r_comfort

    # 纯 progress reward
    total_reward = r_progress

    total_reward = torch.clamp(total_reward, -config.reward_clip, config.reward_clip)

    return {
        "reward": total_reward,
        "reward_components": components,
    }


# 模块级全局变量：running reward baseline
_running_reward_mean = None
_running_reward_alpha = 0.01


def compute_rewards_and_advantages(
    pred_traj: Tensor,
    gt_traj: Tensor,
    neighbor_future: Tensor | None = None,
    config: RewardConfig | None = None,
    group_size: int = 1,
) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """
    标准 GRPO advantage：组内 z-score 归一化，不用 ranking。
    """
    global _running_reward_mean

    result = compute_reward(
        pred_traj=pred_traj,
        gt_traj=gt_traj,
        neighbor_future=neighbor_future,
        config=config,
        group_size=group_size,
    )

    rewards = result["reward"]

    B_total = rewards.shape[0]
    B = B_total // group_size

    rewards_grouped = rewards.view(B, group_size)

    # ============================================================
    # 标准 GRPO: 组内 z-score
    # ============================================================
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True)

    # [关键] 如果组内标准差太小（< 0.01），这组不产生梯度
    # 用 max(std, 0.01) 而不是 std + 1e-8，避免放大噪声
    std_safe = torch.max(std, torch.tensor(0.01, device=std.device))

    advantages = ((rewards_grouped - mean) / std_safe).reshape(-1)

    return rewards, advantages, result["reward_components"]
