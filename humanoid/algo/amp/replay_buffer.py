# SPDX-License-Identifier: BSD-3-Clause
"""Circular replay buffer for policy AMP transition features."""
from __future__ import annotations

import torch


class AMPReplayBuffer:
    """Fixed-capacity circular buffer storing policy amp_obs features.

    The discriminator treats these as the "generated" (fake) samples, paired with expert
    minibatches drawn from the MotionLib.
    """

    def __init__(self, amp_obs_dim: int, capacity: int, device: str = "cpu"):
        self.capacity = int(capacity)
        self.dim = int(amp_obs_dim)
        self.device = device
        self.buf = torch.zeros(self.capacity, self.dim, dtype=torch.float32, device=device)
        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add(self, amp_obs: torch.Tensor):
        """amp_obs: (N, dim) on device."""
        n = amp_obs.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            self.buf.copy_(amp_obs[-self.capacity:].to(self.device))
            self._ptr = 0
            self._size = self.capacity
            return
        idx = (self._ptr + torch.arange(n, device=self.device)) % self.capacity
        self.buf.index_copy_(0, idx, amp_obs.to(self.device))
        self._ptr = int((self._ptr + n) % self.capacity)
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        if self._size == 0:
            return torch.zeros(batch_size, self.dim, device=self.device)
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return self.buf[idx]

    def state_dict(self) -> dict:
        # Only the discriminator weights / its optimizer / the expert-logit EMA carry
        # learned state that *must* survive resume. The policy replay buffer is a
        # transient sampling reservoir that refills every step (num_envs * num_steps
        # transitions per PPO iteration). We therefore do NOT persist its contents:
        # a 1M x 35 float32 tensor would bloat every checkpoint (~140 MB) and, worse,
        # restoring ptr/size over a freshly-allocated zero buffer would feed all-zero
        # "policy" features to the discriminator (silent corruption). size is reported
        # only for logging.
        return {"size": self._size}

    def load_state_dict(self, state: dict):
        # Intentionally reset to empty on resume: no learned state is lost (the buffer
        # holds raw transition features, regenerated each step), checkpoints stay lean,
        # and we avoid the all-zero-sample corruption that restoring stale ptr/size would
        # cause. The buffer refills within a single PPO iteration.
        self._ptr = 0
        self._size = 0
