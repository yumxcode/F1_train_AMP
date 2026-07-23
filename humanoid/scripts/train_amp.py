# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 AgiBot Inc / F1 AMP project.
"""AMP training entrypoint (Isaac Gym required at runtime).

Mirrors ``scripts/train.py`` but defaults the task to the registered AMP task
``x1_amp``. The plain PPO baseline task ``x1_dh_stand`` is left untouched and is
still trainable via the original ``scripts/train.py`` for regression comparison.

Run (remote Gradmotion / Isaac Gym host):
    python humanoid/scripts/train_amp.py --task=x1_amp --headless --num_envs=4096
"""
from humanoid.envs import *                      # noqa: F401,F403  (registers tasks)
from humanoid.utils import get_args, task_registry


def train(args):
    name = args.task if getattr(args, "task", None) else "x1_amp"
    env, env_cfg = task_registry.make_env(name=name, args=args)
    runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=name, args=args)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=False)
    return runner, log_dir


if __name__ == "__main__":
    args = get_args()
    train(args)
