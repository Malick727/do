import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from part_preparation.paths import RESULTS_DIR
from part_preparation.solvers.bilevel.rl import features
from part_preparation.solvers.bilevel.rl.net import ActorCriticNetwork


class Agent:

    def __init__(self, env, hidden_dim=64, lr=1e-3, gamma=1.0, lam=0.95,
                 value_coef=0.5, entropy_coef=0.01):
        self.network = ActorCriticNetwork(
            engine_input_dim=features.ENGINE_FEATURES,
            msn_input_dim=features.MSN_FEATURES,
            factory_input_dim=(features.FACTORY_BASE_FEATURES
                               + features.scales(env.problem).max_capacity),
            state_input_dim=len(features.state_features(env, env.reset())),
            hidden_dim=hidden_dim,
        )
        self.gamma, self.lam = gamma, lam
        self.value_coef, self.entropy_coef = value_coef, entropy_coef
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.actor_params = [p for name, p in self.network.named_parameters()
                             if not name.startswith("value")]
        self.critic_params = [p for name, p in self.network.named_parameters()
                              if name.startswith("value")]
        self.reward_scale = None
        self.logps, self.values, self.entropies, self.rewards = [], [], [], []
        self.losses = []
        self.phase = "train"
        self.episode_log = []

    @staticmethod
    def _pick(score_fn, feats, deterministic):
        logits = score_fn(torch.as_tensor(feats))
        dist = Categorical(logits=logits)
        k = logits.argmax() if deterministic else dist.sample()
        return int(k), dist.log_prob(k), dist.entropy() / np.log(max(len(feats), 2))

    def act(self, env, state, deterministic=False):
        net = self.network

        engine_feats, engine_idx = features.engine_candidates(env, state)
        if len(engine_idx) == 0:
            return None
        k, logp_e, ent_e = self._pick(net.score_engines, engine_feats, deterministic)
        engine_i = int(engine_idx[k])

        msn_feats, msn_idx = features.msn_candidates(env, state, engine_i)
        if len(msn_idx) == 0:
            return None
        k, logp_m, ent_m = self._pick(net.score_msns, msn_feats, deterministic)
        msn_i = int(msn_idx[k])

        factory_feats, factory_idx = features.factory_candidates(env, state, engine_i, msn_i)
        if len(factory_idx) == 0:
            return None
        k, logp_f, ent_f = self._pick(net.score_factories, factory_feats, deterministic)
        factory_i = int(factory_idx[k])

        if not deterministic:
            self.logps.append(logp_e + logp_m + logp_f)
            self.entropies.append(ent_e + ent_m + ent_f)
            self.values.append(net.state_value(
                torch.as_tensor(features.state_features(env, state))))

        return engine_i, msn_i, factory_i

    def set_actor_trainable(self, trainable):
        for p in self.actor_params:
            p.requires_grad_(trainable)

    def gae(self, rewards, values):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            running = delta + self.gamma * self.lam * running
            advantages[t] = running
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def finish_episode(self):
        if not self.rewards:
            return None
        assert len(self.rewards) == len(self.logps), "buffers desalignes"

        if self.reward_scale is None:
            self.reward_scale = max(abs(np.mean(self.rewards)) * len(self.rewards) / 100.0, 1e-8)

        rewards = [r / self.reward_scale for r in self.rewards]
        values = [float(v.detach()) for v in self.values]
        advantages, returns = self.gae(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(torch.stack(self.logps) * torch.as_tensor(advantages)).mean()
        value_loss = F.mse_loss(torch.stack(self.values), torch.as_tensor(returns))
        entropy = torch.stack(self.entropies).mean() / 3.0
        self.losses.append(policy_loss + self.value_coef * value_loss
                           - self.entropy_coef * entropy)

        stats = {"cost": -sum(self.rewards), "entropy": float(entropy.detach()),
                 "value_loss": float(value_loss.detach())}
        self.episode_log.append({
            "episode": len(self.episode_log),
            "phase": self.phase,
            "steps": len(self.rewards),
            "cost": stats["cost"],
            "policy_loss": float(policy_loss.detach()),
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
        })
        self.logps, self.values, self.entropies, self.rewards = [], [], [], []
        return stats

    def update(self):
        if not self.losses:
            return
        self.optimizer.zero_grad()
        torch.stack(self.losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_params, 1.0)
        self.optimizer.step()
        self.losses = []

    def save_logs(self, name):
        columns = {
            "rewards": ["episode", "phase", "steps", "cost"],
            "losses": ["episode", "phase", "policy_loss", "value_loss", "entropy"],
        }
        for kind, fields in columns.items():
            path = RESULTS_DIR / "rl" / kind / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.episode_log)

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
