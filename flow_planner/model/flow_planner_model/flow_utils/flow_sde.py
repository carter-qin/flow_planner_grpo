import math
import torch
from flow_planner.model.model_base import Scheduler


class FlowSDE(Scheduler):
    def __init__(self, path, **sample_params):
        """
        FlowSDE: SDE sampler for Flow-GRPO with log probability tracking.

        Does NOT use VelocityModel wrapper (which does t.unsqueeze(0) for
        ODE solver's scalar t — incompatible with batch t here).
        Instead, directly calls model_fn and applies pred_transform_func inline.

        IMPORTANT: AffineProbPath.target_to_velocity() does NOT call
        expand_tensor_like on t. So when t is (B,), the scheduler outputs
        (B,) alpha_t etc., which cannot broadcast to (B, P, A, D).
        We must reshape t to (B, 1, 1, 1) before passing to velocity_func.
        """
        self.path = path

        self.sample_steps = sample_params.get("sample_steps", 4)
        self.noise_level = sample_params.get("noise_level", 0.7)

        self.translation_funcs = self._get_translation_funcs()

    def _reshape_t_for_broadcast(self, t_batch, x):
        """
        Reshape t from (B,) to (B, 1, 1, ...) to match x's ndim.

        AffineProbPath methods (target_to_velocity, etc.) pass t directly
        into scheduler, which returns (B,)-shaped alpha_t, sigma_t, etc.
        These are then multiplied with x_t (B, P, A, D) without expand.

        ODE path avoids this because VelocityModel passes (1,) scalar t,
        which broadcasts to any shape. We must explicitly reshape for batch t.
        """
        extra_dims = x.ndim - t_batch.ndim
        if extra_dims > 0:
            t_batch = t_batch.reshape(-1, *([1] * extra_dims))
        return t_batch

    def generate(
        self,
        x_init,
        model_fn,
        model_pred_type,
        noise_level=None,
        sample_steps=None,
        **model_extra,
    ):
        """
        SDE rollout with log probability tracking.

        Args:
            x_init: (B, P, action_len, state_dim) initial noise
            model_fn: decoder callable(x, t, **extras) -> prediction
            model_pred_type: "x_start", "velocity", or "noise"
            noise_level: override for self.noise_level
            sample_steps: override for self.sample_steps
            **model_extra: passed to model_fn

        Returns:
            all_x: (B, steps+1, P, action_len, state_dim)
            all_log_probs: (B, steps)
            all_prev_x_means: (B, steps, P, action_len, state_dim)
            all_std_dev_t: (steps,) scalar tensor per step
        """
        steps = sample_steps if sample_steps is not None else self.sample_steps
        sigma = noise_level if noise_level is not None else self.noise_level

        velocity_func = self.translation_funcs[(model_pred_type, "velocity")]

        eps = 1e-3
        timesteps = torch.linspace(eps, 1, steps + 1, device=x_init.device)

        x = x_init

        all_x = [x]
        all_log_probs = []
        all_prev_x_means = []
        all_std_dev_t = []

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr

            # (B,) for decoder's TimestepEmbedder: expects (B,) -> t[:, None]
            t_batch = torch.ones((x.shape[0],), device=x.device) * t_curr

            # Direct model call (bypasses VelocityModel's t.unsqueeze(0))
            pred = model_fn(x, t_batch, **model_extra)

            # Reshape t for AffineProbPath broadcast: (B,) -> (B, 1, 1, 1)
            # AffineProbPath.target_to_velocity does scheduler(t) -> (B,) alpha_t,
            # then a_t * x_t which fails if a_t is (B,) and x_t is (B, P, A, D).
            t_broadcast = self._reshape_t_for_broadcast(t_batch, x)
            v_pred = velocity_func(pred, x, t_broadcast)

            # SDE Step
            x, log_prob, prev_x_mean, std_dev_t = self.step_with_logprob(
                model_output=v_pred,
                sample=x,
                dt=dt,
                t=t_curr.item(),
                noise_level=sigma,
                prev_sample=None,
            )

            all_x.append(x)
            all_log_probs.append(log_prob)
            all_prev_x_means.append(prev_x_mean)
            all_std_dev_t.append(std_dev_t)

        return (
            torch.stack(all_x, dim=1),
            torch.stack(all_log_probs, dim=1),
            torch.stack(all_prev_x_means, dim=1),
            torch.stack(all_std_dev_t, dim=0),  # (steps,)
        )

    def step_with_logprob(
        self, model_output, sample, dt, t, noise_level, prev_sample=None
    ):
        """
        Flow-GRPO SDE Step with Dynamic Diffusion Coefficient.

        Args:
            model_output: velocity prediction (after pred_transform_func)
            sample: current x_t
            dt: time step size (tensor scalar)
            t: current time (python float)
            noise_level: diffusion noise level
            prev_sample: if provided, use this instead of sampling

        Returns:
            prev_sample: next x_{t+dt}
            log_prob: (B,) log probability of the transition
            prev_sample_mean: deterministic drift target
            std_dev_t: scalar tensor, diffusion coefficient at this step
        """
        device = sample.device
        t = max(t, 1e-5)
        t = min(t, 0.999)

        std_dev_t = torch.tensor(noise_level * math.sqrt(t / (1 - t)), device=device)

        diffusion_scale = std_dev_t * torch.sqrt(dt)

        factor = (std_dev_t**2) / (2 * t)
        correction_term = factor * (sample + (1 - t) * model_output)

        drift = model_output + correction_term
        prev_sample_mean = sample + drift * dt

        if prev_sample is None:
            variance_noise = torch.randn_like(model_output)
            prev_sample = prev_sample_mean + diffusion_scale * variance_noise

        variance = diffusion_scale**2
        variance = torch.clamp(variance, min=1e-7)

        log_prob = (
            -0.5 * ((prev_sample.detach() - prev_sample_mean) ** 2) / variance
            - 0.5 * torch.log(variance)
            - 0.5 * math.log(2 * math.pi)
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    def identity(self, x, xt, t):
        return x

    def _get_translation_funcs(self):
        return {
            ("velocity", "x_start"): self.path.velocity_to_target,
            ("velocity", "noise"): self.path.velocity_to_epsilon,
            ("x_start", "velocity"): self.path.target_to_velocity,
            ("x_start", "noise"): self.path.target_to_epsilon,
            ("noise", "velocity"): self.path.epsilon_to_velocity,
            ("noise", "x_start"): self.path.epsilon_to_target,
            ("velocity", "velocity"): self.identity,
            ("x_start", "x_start"): self.identity,
            ("noise", "noise"): self.identity,
        }
