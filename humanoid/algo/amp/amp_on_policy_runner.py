# SPDX-License-Identifier: BSD-3-Clause
"""AMP on-policy runner: extends DHOnPolicyRunner with a discriminator and style reward.

Design goals (see data/amp_contract.md §8):
  * expert/policy features share the env's ``compute_amp_obs`` builder (identical schema);
  * the discriminator is trained once per PPO iteration on fresh expert + policy minibatches;
  * the style reward is added to the env reward BEFORE ``process_env_step`` so GAE sees it;
  * checkpoint save/load include discriminator + its optimizer + expert-logit EMA so resume
    does NOT lose discriminator/optimizer/buffer state.
The plain PPO baseline (``x1_dh_stand``) is untouched.
"""
from __future__ import annotations

import os
import time
import statistics
import torch
from collections import deque
from datetime import datetime

from ..ppo.dh_on_policy_runner import DHOnPolicyRunner
from .motion_lib import MotionLib, AMP_OBS_DIM, AMP_DISC_DIM, AMP_NUM_OBS_STEPS
from .discriminator import Discriminator, compute_disc_loss
from .replay_buffer import AMPReplayBuffer


class AMPOnPolicyRunner(DHOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        amp_cfg = train_cfg.get("amp", {})
        self.style_weight = float(amp_cfg.get("style_weight", 1.0))
        self.style_weight_warmup_iters = int(amp_cfg.get("style_weight_warmup_iters", 500))
        self.style_weight_max = float(amp_cfg.get("style_weight_max", 1.0))
        self.disc_train_iters = int(amp_cfg.get("disc_train_iters", 2))
        self.disc_batch = int(amp_cfg.get("disc_batch_size", 4096))
        self.disc_lr = float(amp_cfg.get("disc_lr", 1e-4))
        self.disc_gp = float(amp_cfg.get("disc_grad_penalty_coef", 5.0))
        self.ema_decay = float(amp_cfg.get("expert_logit_ema_decay", 0.95))

        disc_dim = int(amp_cfg.get("disc_input_dim", AMP_DISC_DIM))  # 350 (10-step stacked)
        disc_hid = amp_cfg.get("disc_hidden_dims", [1024, 512])
        self.disc = Discriminator(disc_dim, hidden_dims=disc_hid).to(self.device)
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.disc_lr)

        # expert motion library (retargeted clips). Empty list => style reward disabled.
        clip_paths = amp_cfg.get("expert_clip_paths", []) or []
        # Resolve the {LEGGED_GYM_ROOT_DIR} placeholder. The asset config uses the same
        # convention and LeggedRobot substitutes it via .format(); MotionLib/np.load do NOT,
        # so an unsubstituted path (BUG, iteration-2 audit) raised FileNotFoundError at runner
        # construction before any training step ran. Substitute here against the real root dir.
        try:
            from humanoid import LEGGED_GYM_ROOT_DIR as _root
        except Exception:
            _root = os.getcwd()
        resolved = []
        for p in clip_paths:
            p = p.replace("{LEGGED_GYM_ROOT_DIR}", str(_root))
            p = os.path.expanduser(p)
            resolved.append(p)
        clip_paths = resolved
        self.motion_lib = MotionLib(clip_paths, target_fps=100.0, device=self.device) if clip_paths else None
        self.policy_amp_buffer = AMPReplayBuffer(disc_dim, int(amp_cfg.get("policy_buffer_capacity", 1_000_000)), device=self.device)
        self.register_buffer_compat = None
        self.expert_logit_ema = torch.tensor(float(amp_cfg.get("expert_logit_ema_init", 0.0)), device=self.device)

        # Policy-side 10-step history buffer for building stacked disc features.
        # compute_amp_obs() returns (N, 35) per step; we stack 10 consecutive steps.
        self.num_obs_steps = AMP_NUM_OBS_STEPS
        self._amp_history = None  # will be initialized as (N, num_steps, 35) on first call

        # smoke-sanity: env must expose the AMP feature builder
        if not hasattr(self.env, "compute_amp_obs"):
            raise RuntimeError("AMP env must implement compute_amp_obs() -> (N, F) tensor")
        with torch.no_grad():
            _probe = self.env.compute_amp_obs()
        assert _probe.shape[-1] == AMP_OBS_DIM, (
            f"env amp_obs dim {_probe.shape[-1]} != single-step AMP_OBS_DIM {AMP_OBS_DIM}; "
            "expert/policy schema mismatch")
        print(f"[AMP] disc_input_dim={disc_dim} (10-step stacked) style_weight={self.style_weight} "
              f"expert_clips={len(clip_paths)} expert_samples={self.motion_lib.num_samples if self.motion_lib else 0}")

    # ------------------------------------------------------------------ #
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            self.it = it
            # AMP style weight curriculum: ramp from 0 to max over warmup iters.
            # This prevents disc saturation when expert/policy distributions are trivially
            # separable at init (single short clip vs random policy). Once the policy learns
            # to walk (via task reward), the distributions overlap and AMP becomes meaningful.
            if self.style_weight_warmup_iters > 0 and it < self.style_weight_warmup_iters:
                effective_style_weight = self.style_weight_max * (it / self.style_weight_warmup_iters)
            else:
                effective_style_weight = self.style_weight_max
            start = time.time()
            # The inherited log() reads locs['obs_mean']/locs['obs_std'] (parent logs the
            # first num_single_obs=47 entries). The base DHOnPolicyRunner computes these before
            # the rollout loop; the original AMP override DROPPED this line, so self.log(locals())
            # crashed with KeyError on the very first iteration whenever log_dir was set (smoke +
            # real training). Restored here for parity with the parent.
            obs_std, obs_mean = torch.std_mean(obs, dim=0)
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (obs.to(self.device), critic_obs.to(self.device),
                                                       rewards.to(self.device), dones.to(self.device))
                    # === AMP: add style reward from discriminator over policy amp_obs ===
                    rewards = self._add_style_reward(rewards, effective_style_weight)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_state_estimator_loss = self.alg.update()
            # === AMP: train discriminator only after warmup (gate disc training to style_weight curriculum) ===
            # During warmup (style_weight=0), disc is NOT trained. This prevents saturation
            # when expert/policy distributions are trivially separable at init.
            if effective_style_weight > 0.01:
                disc_metrics = self._train_discriminator()
            else:
                disc_metrics = self._last_disc_metrics  # reuse last (or initial zeros)
            learn_time = time.time() - start
            if self.log_dir is not None:
                self.log(locals())          # PPO console + tensorboard logging (parent)
                self._log_amp(locals(), disc_metrics)  # AMP-specific scalars
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    # ------------------------------------------------------------------ #
    def _build_stacked_amp(self, single_step: torch.Tensor) -> torch.Tensor:
        """Build 10-step stacked AMP feature from single-step (N, 35) → (N, 350).
        
        Maintains a rolling history buffer of the last 10 steps.
        """
        N = single_step.shape[0]
        if self._amp_history is None:
            # Initialize history with zeros
            self._amp_history = torch.zeros(N, self.num_obs_steps, AMP_OBS_DIM, device=self.device)
        # Shift history: drop oldest, append newest
        self._amp_history = torch.cat([
            self._amp_history[:, 1:, :],
            single_step.unsqueeze(1)
        ], dim=1)
        # Flatten to (N, 350)
        return self._amp_history.reshape(N, -1)

    @torch.no_grad()
    def _add_style_reward(self, rewards: torch.Tensor, sw: float = None) -> torch.Tensor:
        if self.motion_lib is None:
            return rewards
        if sw is None:
            sw = self.style_weight
        policy_amp = self.env.compute_amp_obs()  # (N, 35)
        stacked = self._build_stacked_amp(policy_amp)  # (N, 350)
        self.policy_amp_buffer.add(stacked)
        d_p = self.disc(stacked)
        style_r = self.disc.compute_reward(d_p, self.expert_logit_ema)
        return rewards + sw * style_r

    def _train_discriminator(self):
        if self.motion_lib is None or self.policy_amp_buffer.size == 0:
            return {"disc_loss": 0.0, "disc_acc": 0.0, "expert_logit": 0.0,
                    "policy_logit": 0.0, "style_reward": 0.0}
        metrics_acc = {"loss": 0.0, "expert_loss": 0.0, "policy_loss": 0.0,
                       "grad_penalty": 0.0, "accuracy": 0.0, "expert_logit": 0.0, "policy_logit": 0.0}
        for _ in range(self.disc_train_iters):
            expert = self.motion_lib.sample_expert(self.disc_batch)
            policy = self.policy_amp_buffer.sample(self.disc_batch)
            m = compute_disc_loss(self.disc, expert, policy, grad_penalty_coef=self.disc_gp)
            self.disc_optimizer.zero_grad()
            m["loss"].backward()
            self.disc_optimizer.step()
            for k in metrics_acc:
                metrics_acc[k] += m[k].item()
        n = max(self.disc_train_iters, 1)
        for k in metrics_acc:
            metrics_acc[k] /= n
        # update expert logit EMA for the reward transform
        with torch.no_grad():
            self.expert_logit_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * metrics_acc["expert_logit"])
        with torch.no_grad():
            d_p = self.disc(self.policy_amp_buffer.sample(min(self.disc_batch, max(self.policy_amp_buffer.size, 1))))
            metrics_acc["style_reward"] = float(self.disc.compute_reward(d_p, self.expert_logit_ema).mean())
        return metrics_acc

    def _log_amp(self, locs, dm):
        # PPO logs already happen in parent log(); here we log AMP-specific scalars.
        if self.writer is not None:
            for k, v in dm.items():
                self.writer.add_scalar("AMP/" + k, v, locs["it"])
            self.writer.add_scalar("AMP/expert_logit_ema", float(self.expert_logit_ema), locs["it"])
            self.writer.add_scalar("AMP/policy_buffer_size", self.policy_amp_buffer.size, locs["it"])
        # Console summary so remote smoke/train logs surface the INDEPENDENT AMP metrics
        # (discriminator loss, gradient penalty, style reward) needed as Gate-A evidence.
        print(f"  [AMP] iter={locs['it']} disc_loss={dm['loss']:.4f} gp={dm['grad_penalty']:.4f} "
              f"style_r={dm['style_reward']:.4f} acc={dm['accuracy']:.3f} "
              f"e_logit={dm['expert_logit']:.3f} p_logit={dm['policy_logit']:.3f} "
              f"ema={float(self.expert_logit_ema):.3f}", flush=True)

    # ------------------------------------------------------------------ #
    def save(self, path, infos=None):
        torch.save({
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "es_optimizer_state_dict": self.alg.state_estimator_optimizer.state_dict(),
            "disc_state_dict": self.disc.state_dict(),
            "disc_optimizer_state_dict": self.disc_optimizer.state_dict(),
            "expert_logit_ema": float(self.expert_logit_ema),
            "policy_amp_buffer_state": self.policy_amp_buffer.state_dict(),
            "amp_style_weight": self.style_weight,
            "iter": self.it,
            "infos": infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.state_estimator_optimizer.load_state_dict(loaded_dict["es_optimizer_state_dict"])
        # AMP-specific state (resume-safe); tolerate legacy checkpoints without these keys
        if "disc_state_dict" in loaded_dict:
            self.disc.load_state_dict(loaded_dict["disc_state_dict"])
        if load_optimizer and "disc_optimizer_state_dict" in loaded_dict:
            self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
        if "expert_logit_ema" in loaded_dict:
            # Rebuild the EMA tensor non-inplace and OUTSIDE inference_mode: .fill_() on
            # a tensor created inside InferenceMode (the __init__ default when play.py /
            # resume loads a checkpoint under the gymtorch-enabled global inference mode)
            # raises "Inplace update to inference tensor outside InferenceMode" under
            # PyTorch>=2.x. torch.inference_mode(False) yields a normal autograd tensor.
            with torch.inference_mode(False):
                self.expert_logit_ema = torch.tensor(
                    float(loaded_dict["expert_logit_ema"]),
                    dtype=self.expert_logit_ema.dtype,
                    device=self.expert_logit_ema.device)
        if "policy_amp_buffer_state" in loaded_dict:
            self.policy_amp_buffer.load_state_dict(loaded_dict["policy_amp_buffer_state"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
