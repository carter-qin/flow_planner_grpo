from __future__ import annotations
import logging
import os
import time
from typing import Dict, Optional
import argparse
import random
import copy
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_planner.train_utils import ddp
from flow_planner.data.utils.collect import collect_batch
from flow_planner.train_utils.ddp import reduce_and_average_losses, ddp_setup_universal
from flow_planner.train_utils.save_model import save_model_lora, resume_model
from flow_planner.model.model_utils.lora import (
    freeze_non_lora_params,
    count_trainable_params,
)
from flow_planner.model.rewards import (
    RewardConfig,
    compute_rewards_and_advantages,
)


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_log_prob_for_step(
    model,
    flow_sde,
    sample_dict,
    step_idx,
    decoder_model_extra,
    model_pred_type,
):
    """
    Recompute log_prob for a single SDE step with gradients.

    CRITICAL: must apply the same pred_transform_func as rollout phase
    to convert raw model prediction to velocity before passing to step_with_logprob.
    Also must reshape t for AffineProbPath broadcast compatibility.
    """
    x_t = sample_dict["samples"][:, step_idx]
    x_next = sample_dict["next_samples"][:, step_idx]

    t_curr = sample_dict["timesteps"][step_idx]
    t_next = sample_dict["timesteps"][step_idx + 1]
    dt = t_next - t_curr

    B_total = x_t.shape[0]
    t_batch = torch.ones((B_total,), device=x_t.device) * t_curr

    # Raw model prediction (e.g. x_start for model_pred_type="x_start")
    pred = model.decoder(x_t, t_batch, **decoder_model_extra)

    # Convert to velocity — must match what flow_sde.generate() does
    # Reshape t for AffineProbPath broadcast: (B,) -> (B, 1, 1, 1)
    velocity_func = flow_sde.translation_funcs[(model_pred_type, "velocity")]
    t_broadcast = flow_sde._reshape_t_for_broadcast(t_batch, x_t)
    v_pred = velocity_func(pred, x_t, t_broadcast)

    _, log_prob, prev_sample_mean, std_dev_t = flow_sde.step_with_logprob(
        model_output=v_pred,
        sample=x_t,
        dt=dt,
        t=t_curr.item(),
        noise_level=flow_sde.noise_level,
        prev_sample=x_next,
    )

    return log_prob, prev_sample_mean, std_dev_t


def find_resume_ckpt(cfg) -> str | None:
    """
    Determine if we should resume from an RL checkpoint.
    Returns the path to the checkpoint directory, or None.
    """
    resume_path = cfg.get("resume_path", None)
    should_resume = cfg.get("should_resume", False)

    if resume_path is not None:
        ckpt_file = (
            resume_path
            if resume_path.endswith(".pth")
            else os.path.join(resume_path, "latest.pth")
        )
        if os.path.isfile(ckpt_file):
            return resume_path
        else:
            print(f"[WARNING] resume_path specified but file not found: {ckpt_file}")
            return None

    if should_resume:
        latest_ckpt = os.path.join(cfg.save_dir, "latest.pth")
        if os.path.isfile(latest_ckpt):
            return cfg.save_dir

    return None


def load_sft_checkpoint(model, ckpt_path, logger):
    """
    Load pretrained SFT checkpoint into model.
    LoRA parameters will be missing (expected) since SFT model doesn't have them.
    """
    logger.info(f"Loading SFT checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=True)

    pretrained_sd = {
        n.replace("module.", ""): v for n, v in ckpt["ema_state_dict"].items()
    }

    missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)

    non_lora_missing = [k for k in missing if "lora" not in k.lower()]
    if non_lora_missing:
        logger.warning(
            f"Non-LoRA keys missing from SFT checkpoint (possible architecture mismatch): "
            f"{non_lora_missing}"
        )

    lora_missing = [k for k in missing if "lora" in k.lower()]
    logger.info(
        f"SFT checkpoint loaded. "
        f"LoRA keys initialized randomly: {len(lora_missing)} | "
        f"Non-LoRA missing: {len(non_lora_missing)} | "
        f"Unexpected: {len(unexpected)}"
    )

    if unexpected:
        logger.warning(f"Unexpected keys in SFT checkpoint: {unexpected}")

    return model


@hydra.main(version_base=None, config_path="script")
def trainer_rl(cfg: DictConfig):

    set_seed(cfg.seed)

    global_rank, local_rank, world_size = ddp_setup_universal(verbose=True, cfg=cfg)

    if local_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    log_path = os.path.join(cfg.save_dir, f"{cfg.job_name}.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    assert cfg.train.batch_size >= world_size, (
        f"batch size is at least world size, but got batch size {cfg.train.batch_size} running on {world_size} devices."
    )

    if global_rank > 0:
        logger.setLevel(logging.WARNING)

    # ===================================================
    # Build model (with LoRA in decoder)
    # ===================================================
    logger.info("build model")
    model = instantiate(cfg.model)

    # ===================================================
    # Weight loading: resume RL > load SFT > train from scratch
    # ===================================================
    resume_ckpt_path = find_resume_ckpt(cfg)
    init_epoch = 0
    wandb_id = None

    freeze_non_lora_params(model)
    param_stats = count_trainable_params(model)
    logger.info(
        f"LoRA enabled | Total params: {param_stats['total']:,} | "
        f"Trainable params: {param_stats['trainable']:,} | "
        f"Ratio: {param_stats['ratio']}"
    )

    # Build optimizer
    logger.info("build optimizer (LoRA params only)")
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = instantiate(
        cfg.optimizer,
        params=[{"params": lora_params}],
        lr=cfg.train.rl_lr,
    )

    logger.info("build scheduler")
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    logger.info("build ema")
    ema = instantiate(cfg.ema, model=model)

    if resume_ckpt_path is not None:
        logger.info(f"Resuming RL training from {resume_ckpt_path}")
        model, optimizer, scheduler, init_epoch, wandb_id, ema = resume_model(
            resume_ckpt_path, model, optimizer, scheduler, ema, cfg.device
        )
        freeze_non_lora_params(model)
        logger.info(f"Resumed from epoch {init_epoch}")

    elif cfg.get("pretrained_checkpoint", None) is not None:
        model = load_sft_checkpoint(model, cfg.pretrained_checkpoint, logger)
        freeze_non_lora_params(model)

    else:
        logger.warning("No checkpoint provided. Training from scratch.")

    # ===================================================
    # Build reference model (frozen copy for KL penalty)
    # ===================================================
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model = ref_model.to(local_rank)

    model = model.to(local_rank)
    if cfg.ddp.distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=global_rank,
            find_unused_parameters=True,
        )

    # construct dataset & dataloader
    logger.info("construct dataset and dataloader")
    trainset = instantiate(cfg.data.dataset.train)
    trainsampler = DistributedSampler(
        trainset, num_replicas=world_size, rank=global_rank, shuffle=True
    )
    trainloader = DataLoader(
        trainset,
        sampler=trainsampler,
        batch_size=cfg.train.batch_size // world_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
        collate_fn=collect_batch,
    )

    lora_params = [p for p in ddp.get_model(model).parameters() if p.requires_grad]

    # initialize recorder
    logger.info("initialize recorder")
    recorder_dict = {}
    if hasattr(cfg, "recorder"):
        for name, rec in cfg.recorder.items():
            if name == "wandb":
                recorder_dict[name] = instantiate(rec, wandb_id=wandb_id)
            else:
                recorder_dict[name] = instantiate(rec)

    # ===================================================
    # RL Training Loop
    # ===================================================
    logger.info("RL Training launched (LoRA fine-tuning)")
    group_size = cfg.train.get("group_size", 8)
    reward_config = RewardConfig(dt=0.1)

    flow_sde = ddp.get_model(model).flow_sde
    model_pred_type = ddp.get_model(model).model_type
    num_train_timesteps = flow_sde.sample_steps

    timer = time.time()

    try:
        with tqdm(
            total=cfg.train.epoch, initial=init_epoch, disable=(global_rank != 0)
        ) as epoch_bar:
            for epoch in range(init_epoch, cfg.train.epoch):
                trainsampler.set_epoch(epoch)
                model.eval()

                loss_list = []
                reward_list = []
                kl_list = []
                reward_comp_accum: Dict[str, list] = defaultdict(list)

                # [新增] 每 100 步用前 3 个 batch 做 eval rollout，监控固定场景的 reward 变化
                eval_data_cache = []

                with tqdm(
                    total=len(trainloader),
                    desc=f"Epoch {epoch + 1} - RL Training",
                    disable=True,  # 关掉 tqdm 进度条，避免和 print 混在一起
                    leave=False,
                ) as batch_bar:
                    for k, data in enumerate(trainloader):
                        data = data.to(cfg.device)
                        policy_model = ddp.get_model(model)

                        # [新增] 缓存前 3 个 batch 用于固定 eval
                        if len(eval_data_cache) < 3:
                            eval_data_cache.append(
                                data.clone()
                                if hasattr(data, "clone")
                                else copy.deepcopy(data)
                            )

                        # [新增] 每 100 步在固定场景上 eval
                        if (
                            global_rank == 0
                            and k > 0
                            and k % 100 == 0
                            and eval_data_cache
                        ):
                            # [新增] 第一次 eval 时，同时评估 SFT baseline
                            if k == 100:
                                from flow_planner.eval_sft_baseline import (
                                    eval_sft_baseline,
                                )

                                sft_reward = eval_sft_baseline(
                                    ddp.get_model(model), eval_data_cache, cfg.device
                                )
                                print(
                                    f"  [SFT BASELINE] fixed_scene_reward={sft_reward:.4f}",
                                    flush=True,
                                )

                            policy_model.eval()
                            with torch.no_grad():
                                eval_rewards = []
                                for eval_data in eval_data_cache:
                                    eval_data_dev = eval_data.to(cfg.device)
                                    (eval_samples, _, _, _, _, _) = policy_model(
                                        eval_data_dev,
                                        mode="rollout",
                                        group_size=4,
                                    )
                                    gt_3d = eval_data_dev.ego_future[:, :, :3]
                                    gt_x, gt_y, gt_h = (
                                        gt_3d[..., 0:1],
                                        gt_3d[..., 1:2],
                                        gt_3d[..., 2:3],
                                    )
                                    gt_4d = torch.cat(
                                        [gt_x, gt_y, torch.cos(gt_h), torch.sin(gt_h)],
                                        dim=-1,
                                    )
                                    gt_exp = gt_4d.repeat_interleave(4, dim=0)
                                    nf = None
                                    if (
                                        hasattr(
                                            eval_data_dev, "neighbor_future_observed"
                                        )
                                        and eval_data_dev.neighbor_future_observed
                                        is not None
                                    ):
                                        nf = eval_data_dev.neighbor_future_observed.repeat_interleave(
                                            4, dim=0
                                        )
                                    r, _, _ = compute_rewards_and_advantages(
                                        eval_samples, gt_exp, nf, reward_config, 4
                                    )
                                    eval_rewards.append(r.mean().item())
                                avg_eval_rew = sum(eval_rewards) / len(eval_rewards)
                                print(
                                    f"  [EVAL@step{k}] fixed_scene_reward={avg_eval_rew:.4f}",
                                    flush=True,
                                )

                        # =============================================
                        # Phase 1: Rollout (No Grad)
                        # decoder_extra_expanded is returned to reuse
                        # in Phase 2, avoiding a redundant encoder fwd
                        # =============================================
                        policy_model.eval()
                        with torch.no_grad():
                            (
                                samples,
                                all_samples,
                                all_log_probs,
                                all_prev_means,
                                all_std_devs,
                                decoder_extra_expanded,
                            ) = policy_model(
                                data,
                                mode="rollout",
                                group_size=group_size,
                            )

                            # 1. 提取 GT (原始数据是 3 维物理坐标: [x, y, heading])
                            gt_traj_3d = data.ego_future[:, :, :3]  # (B, T, 3)

                            # 2. 将 GT 转为 4 维 [x, y, cos, sin]，对齐 samples 的格式
                            gt_x = gt_traj_3d[..., 0:1]
                            gt_y = gt_traj_3d[..., 1:2]
                            gt_h = gt_traj_3d[..., 2:3]
                            gt_traj_4d = torch.cat(
                                [gt_x, gt_y, torch.cos(gt_h), torch.sin(gt_h)], dim=-1
                            )  # (B, T, 4)

                            # 3. 按 group_size 展开
                            gt_traj_expanded = gt_traj_4d.repeat_interleave(
                                group_size, dim=0
                            )  # (B*G, T, 4)

                            neighbor_future = None
                            if (
                                hasattr(data, "neighbor_future_observed")
                                and data.neighbor_future_observed is not None
                            ):
                                neighbor_future = (
                                    data.neighbor_future_observed.repeat_interleave(
                                        group_size, dim=0
                                    )
                                )  # (B*G, N, T, D)

                            rewards, advantages, reward_components = (
                                compute_rewards_and_advantages(
                                    pred_traj=samples,
                                    gt_traj=gt_traj_expanded,
                                    neighbor_future=neighbor_future,
                                    config=reward_config,
                                    group_size=group_size,
                                )
                            )

                        # Track per-component rewards for this batch
                        for comp_name, comp_val in reward_components.items():
                            reward_comp_accum[comp_name].append(comp_val.mean().item())

                        # =============================================
                        # Prepare sample_dict
                        # (decoder_extra_expanded already from Phase 1)
                        # =============================================
                        eps = 1e-3
                        timesteps = torch.linspace(
                            eps, 1, num_train_timesteps + 1, device=cfg.device
                        )

                        sample_dict = {
                            "samples": all_samples[:, :-1],
                            "next_samples": all_samples[:, 1:],
                            "log_probs": all_log_probs,
                            "timesteps": timesteps,
                            "std_devs": all_std_devs,
                            "advantages": advantages,
                        }

                        # [新增] 组内 reward 方差诊断 — 验证 advantage 是否有信号
                        if global_rank == 0 and k < 10:
                            rewards_grouped = rewards.view(-1, group_size)
                            intra_std = rewards_grouped.std(dim=1).mean().item()
                            inter_std = rewards_grouped.mean(dim=1).std().item()
                            adv_abs = advantages.abs().mean().item()
                            adv_std = advantages.std().item()

                            # [新增] 轨迹多样性：组内 ADE 差异
                            samples_grouped = samples.view(
                                -1, group_size, samples.shape[1], samples.shape[2]
                            )
                            group_mean = samples_grouped.mean(dim=1, keepdim=True)
                            traj_diversity = (
                                (samples_grouped - group_mean)
                                .norm(dim=-1)
                                .mean()
                                .item()
                            )

                            print(
                                f"  [DIAG] intra_rew_std={intra_std:.4f} "
                                f"inter_rew_std={inter_std:.4f} "
                                f"adv_abs={adv_abs:.4f} "
                                f"traj_div={traj_diversity:.4f} "
                                f"rew_range=[{rewards.min():.2f}, {rewards.max():.2f}]",
                                flush=True,
                            )

                        # =============================================
                        # Phase 2: Multi-epoch PPO update on same rollout
                        # =============================================
                        ppo_epochs = cfg.train.get("ppo_epochs", 4)

                        all_ppo_info = defaultdict(list)
                        loss = torch.tensor(0.0, device=cfg.device)

                        # [新增] KL early stopping 阈值
                        max_kl_per_iter = cfg.train.get("max_kl_per_iter", 0.05)

                        for ppo_iter in range(ppo_epochs):
                            optimizer.zero_grad()

                            total_policy_loss = torch.tensor(0.0, device=cfg.device)
                            total_kl_loss = torch.tensor(0.0, device=cfg.device)

                            # [新增] 追踪每个 ppo_iter 的 ratio 统计
                            iter_ratios = []

                            for j in range(num_train_timesteps):
                                log_prob, prev_sample_mean, std_dev_t = (
                                    compute_log_prob_for_step(
                                        model=policy_model,
                                        flow_sde=flow_sde,
                                        sample_dict=sample_dict,
                                        step_idx=j,
                                        decoder_model_extra=decoder_extra_expanded,
                                        model_pred_type=model_pred_type,
                                    )
                                )

                                prev_sample_mean_ref = None
                                if cfg.train.beta > 0:
                                    with torch.no_grad():
                                        _, prev_sample_mean_ref, _ = (
                                            compute_log_prob_for_step(
                                                model=ref_model,
                                                flow_sde=flow_sde,
                                                sample_dict=sample_dict,
                                                step_idx=j,
                                                decoder_model_extra=decoder_extra_expanded,
                                                model_pred_type=model_pred_type,
                                            )
                                        )

                                adv = torch.clamp(
                                    sample_dict["advantages"],
                                    -cfg.train.adv_clip_max,
                                    cfg.train.adv_clip_max,
                                )

                                adv_scale = cfg.train.get("adv_scale", 1.0)
                                adv = adv * adv_scale

                                ratio = torch.exp(
                                    log_prob - sample_dict["log_probs"][:, j]
                                )

                                # [新增] 记录 ratio 统计
                                iter_ratios.append(ratio.detach())

                                unclipped_loss = -adv * ratio
                                clipped_loss = -adv * torch.clamp(
                                    ratio,
                                    1.0 - cfg.train.clip_range,
                                    1.0 + cfg.train.clip_range,
                                )
                                policy_loss = torch.mean(
                                    torch.maximum(unclipped_loss, clipped_loss)
                                )

                                if (
                                    cfg.train.beta > 0
                                    and prev_sample_mean_ref is not None
                                ):
                                    variance = torch.clamp(std_dev_t**2, min=1e-7)
                                    kl_loss = (
                                        (prev_sample_mean - prev_sample_mean_ref) ** 2
                                    ).mean(
                                        dim=tuple(range(1, prev_sample_mean.ndim))
                                    ) / (2 * variance)
                                    kl_loss = kl_loss.mean()
                                    total_kl_loss = total_kl_loss + kl_loss
                                else:
                                    kl_loss = torch.tensor(0.0, device=cfg.device)

                                total_policy_loss = total_policy_loss + policy_loss

                                all_ppo_info["approx_kl"].append(
                                    0.5
                                    * torch.mean(
                                        (log_prob - sample_dict["log_probs"][:, j]) ** 2
                                    )
                                )
                                all_ppo_info["clipfrac"].append(
                                    torch.mean(
                                        (
                                            torch.abs(ratio - 1.0)
                                            > cfg.train.clip_range
                                        ).float()
                                    )
                                )
                                all_ppo_info["policy_loss"].append(policy_loss)
                                if cfg.train.beta > 0:
                                    all_ppo_info["kl_loss"].append(kl_loss)

                            # [新增] 打印每个 ppo_iter 的 ratio 统计
                            if global_rank == 0 and k < 5:
                                all_ratios = torch.cat(iter_ratios)
                                ratio_mean = all_ratios.mean().item()
                                ratio_std = all_ratios.std().item()
                                ratio_min = all_ratios.min().item()
                                ratio_max = all_ratios.max().item()
                                print(
                                    f"    [PPO iter {ppo_iter}] ratio: "
                                    f"mean={ratio_mean:.6f} std={ratio_std:.6f} "
                                    f"min={ratio_min:.4f} max={ratio_max:.4f}",
                                    flush=True,
                                )

                            avg_policy_loss = total_policy_loss / num_train_timesteps
                            avg_kl_loss = total_kl_loss / num_train_timesteps
                            loss = avg_policy_loss + cfg.train.beta * avg_kl_loss

                            loss.backward()

                            # [新增] 梯度监控 - 确认 LoRA 参数有梯度
                            if global_rank == 0 and k < 3 and ppo_iter == 0:
                                grad_norms = []
                                for name, p in ddp.get_model(model).named_parameters():
                                    if p.requires_grad and p.grad is not None:
                                        grad_norms.append((name, p.grad.norm().item()))
                                if grad_norms:
                                    max_grad = max(grad_norms, key=lambda x: x[1])
                                    avg_grad = sum(g for _, g in grad_norms) / len(
                                        grad_norms
                                    )
                                    print(
                                        f"  [GRAD] avg={avg_grad:.8f} "
                                        f"max={max_grad[1]:.8f} ({max_grad[0]}) "
                                        f"n_params={len(grad_norms)}",
                                        flush=True,
                                    )

                            nn.utils.clip_grad_norm_(lora_params, 5.0)
                            optimizer.step()

                            # [新增] KL early stopping：如果 ratio 偏移太大，停止本 batch 的 PPO 迭代
                            with torch.no_grad():
                                all_ratios_cat = torch.cat(iter_ratios)
                                safe_log_ratio = torch.log(
                                    torch.clamp(all_ratios_cat, min=1e-6)
                                )
                                approx_kl_this_iter = (
                                    0.5 * torch.mean(safe_log_ratio**2).item()
                                )
                            if (
                                approx_kl_this_iter > max_kl_per_iter
                                and ppo_iter < ppo_epochs - 1
                            ):
                                if global_rank == 0 and k < 10:
                                    print(
                                        f"    [EARLY STOP] ppo_iter={ppo_iter}, "
                                        f"approx_kl={approx_kl_this_iter:.4f} > {max_kl_per_iter}",
                                        flush=True,
                                    )
                                break

                        all_ppo_info["loss"].append(loss)

                        ema.update(model)

                        # Average over all ppo_epochs * timesteps
                        avg_info = {
                            k: torch.mean(torch.stack(v)).item()
                            for k, v in all_ppo_info.items()
                        }
                        loss_list.append(avg_info["policy_loss"])
                        reward_list.append(rewards.mean().item())
                        kl_list.append(avg_info.get("kl_loss", 0.0))

                        # Build tqdm postfix with reward breakdown
                        tqdm_dict = {
                            "loss": f"{avg_info['policy_loss']:.6f}",
                            "rew": f"{rewards.mean().item():.2f}",
                            "kl": f"{avg_info.get('kl_loss', 0):.6f}",
                            "clip": f"{avg_info['clipfrac']:.4f}",
                        }
                        _abbrev = {
                            "collision": "col",
                            "collision_gate": "gate",
                            "progress": "pro",
                            "heading": "hdg",
                            "smoothness": "smo",
                            "comfort": "cmf",
                        }
                        for comp_name, comp_val in reward_components.items():
                            short = _abbrev.get(comp_name, comp_name[:3])
                            tqdm_dict[short] = f"{comp_val.mean().item():.2f}"

                        batch_bar.update(1)

                        # 逐 batch 打印详细信息 (仅 rank 0)
                        if global_rank == 0:
                            now = time.time()
                            elapsed = now - timer
                            steps_done = k + 1
                            total_steps = len(trainloader)

                            # 全局平均速率（最稳定的估计）
                            rate = steps_done / elapsed if elapsed > 0 else 0

                            remaining = (
                                (total_steps - steps_done) / rate if rate > 0 else 0
                            )

                            def fmt_time(seconds):
                                """Format seconds to HH:MM:SS, supporting >24h."""
                                seconds = int(seconds)
                                h = seconds // 3600
                                m = (seconds % 3600) // 60
                                s = seconds % 60
                                return f"{h:02d}:{m:02d}:{s:02d}"

                            elapsed_str = fmt_time(elapsed)
                            remaining_str = fmt_time(remaining)
                            it_s = f"{rate:.2f}it/s" if rate > 0 else "?it/s"

                            comp_str = " | ".join(
                                f"{k}={v}" for k, v in tqdm_dict.items()
                            )
                            step_str = (
                                f"[Epoch {epoch + 1} Step {steps_done}/{total_steps}] "
                                f"[{elapsed_str}<{remaining_str}, {it_s}] "
                                f"{comp_str}"
                            )
                            print(step_str, flush=True)

                scheduler.step()

                # =============================================
                # Epoch-level aggregation
                # =============================================
                avg_loss = sum(loss_list) / len(loss_list)
                avg_reward = sum(reward_list) / len(reward_list)
                avg_kl = sum(kl_list) / len(kl_list)

                epoch_lr = {
                    f"lr/group_{i}": param_group["lr"]
                    for i, param_group in enumerate(optimizer.param_groups)
                }

                metrics = {
                    "policy_loss": torch.tensor(avg_loss, device=cfg.device),
                    "reward": torch.tensor(avg_reward, device=cfg.device),
                    "approx_kl": torch.tensor(avg_kl, device=cfg.device),
                }

                reward_comp_metrics = {}
                for comp_name, comp_vals in reward_comp_accum.items():
                    avg_val = sum(comp_vals) / len(comp_vals)
                    reward_comp_metrics[f"reward/{comp_name}"] = torch.tensor(
                        avg_val, device=cfg.device
                    )

                all_metrics = {**metrics, **reward_comp_metrics}
                if cfg.ddp.distributed:
                    all_metrics = reduce_and_average_losses(
                        all_metrics, torch.device(cfg.device)
                    )

                metrics = {
                    k: v for k, v in all_metrics.items() if not k.startswith("reward/")
                }
                reward_comp_metrics = {
                    k: v for k, v in all_metrics.items() if k.startswith("reward/")
                }

                reward_breakdown_str = " | ".join(
                    f"{k.split('/')[-1]}: {v:.3f}"
                    for k, v in reward_comp_metrics.items()
                )
                logger.info(
                    f"Epoch {epoch + 1} | "
                    f"Loss: {metrics['policy_loss']:.3e} | "
                    f"Reward: {metrics['reward']:.3f} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"Reward breakdown: [{reward_breakdown_str}]"
                )

                if global_rank == 0:
                    for recorder in recorder_dict.values():
                        recorder.record_loss(
                            {**metrics, **reward_comp_metrics, **epoch_lr},
                            epoch + 1,
                        )

                if global_rank == 0 and (epoch + 1) % cfg.train.save_utd == 0:
                    if "wandb" in recorder_dict.keys():
                        wandb_id = recorder_dict["wandb"].id
                    else:
                        wandb_id = None
                    save_model_lora(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        save_path=cfg.save_dir,
                        epoch=epoch,
                        train_loss=float(metrics["policy_loss"]),
                        wandb_id=wandb_id,
                        ema=ema.ema,
                        save_every_epoch=cfg.save_every_since,
                    )
                    print(f"Model saved in {cfg.save_dir}\n")

                epoch_bar.update(1)

                if cfg.ddp.distributed:
                    torch.cuda.synchronize()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C). Stopping...")
        if global_rank == 0:
            print("\n[Force Exit] Detected Ctrl+C. Killing process immediately.")
        os._exit(1)

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        if global_rank == 0:
            import traceback

            traceback.print_exc()
        os._exit(1)

    finally:
        if global_rank == 0:
            logger.info("Closing recorders...")
            for name, recorder in recorder_dict.items():
                if hasattr(recorder, "close"):
                    try:
                        recorder.close()
                        logger.info(f"Recorder {name} closed.")
                    except Exception as e:
                        logger.error(f"Failed to close recorder {name}: {e}")

        logger.info(
            f"Training finished - Time consumed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - timer))}"
        )

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        os._exit(0)


if __name__ == "__main__":
    trainer_rl()
