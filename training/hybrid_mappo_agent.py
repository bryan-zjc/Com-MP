import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _orthogonal_init(layer, gain=math.sqrt(2.0)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)


class ValueNormalizer:
    """Running return normalization used by MAPPO's centralized critic."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = values.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, values):
        return (values - self.mean) / math.sqrt(self.var + 1e-8)

    def denormalize(self, values):
        return values * math.sqrt(self.var + 1e-8) + self.mean

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])


class ActorCritic(nn.Module):
    def __init__(self, obs_size, alpha_bins, hidden_size):
        super().__init__()
        action_size = len(alpha_bins)
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size * 3, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.apply(_orthogonal_init)
        _orthogonal_init(self.actor[-1], gain=0.01)
        _orthogonal_init(self.critic[-1], gain=1.0)

    def distribution(self, obs, action_mask):
        logits = self.actor(obs)
        logits = logits.masked_fill(~action_mask, -1e9)
        return Categorical(logits=logits)

    def value(self, obs, global_context):
        return self.critic(torch.cat((obs, global_context), dim=-1)).squeeze(-1)


class AsyncMAPPOAgent:
    """Shared actor with a centralized value function for asynchronous TLS decisions."""

    def __init__(self, obs_size, alpha_bins, config, device):
        self.obs_size = obs_size
        self.alpha_bins = np.asarray(alpha_bins, dtype=np.float32)
        self.config = config
        self.device = device
        self.network = ActorCritic(obs_size, self.alpha_bins,
                                   config.hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=config.learning_rate,
                                          eps=1e-5)
        self.value_normalizer = ValueNormalizer()
        self.trajectories = defaultdict(list)

    def act(self, obs, global_context, action_mask, phase_by_action,
            deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        context_t = torch.as_tensor(global_context, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool,
                                 device=self.device).unsqueeze(0)
        phase_map_t = torch.as_tensor(phase_by_action, dtype=torch.long,
                                      device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.network.distribution(obs_t, mask_t)
            if deterministic:
                group_probs = self._group_probabilities(dist.probs, phase_map_t)
                selected_phase = torch.argmax(group_probs, dim=-1)
                candidates = phase_map_t == selected_phase.unsqueeze(-1)
                candidate_probs = dist.probs.masked_fill(~candidates, -1.0)
                action = torch.argmax(candidate_probs, dim=-1)
            else:
                action = dist.sample()
            log_prob = self._group_log_prob(dist.probs, phase_map_t, action)
            value_normalized = self.network.value(obs_t, context_t)
            value = self.value_normalizer.denormalize(value_normalized)
        idx = int(action.item())
        return idx, float(self.alpha_bins[idx]), float(log_prob.item()), float(value.item())

    def add_transition(self, tls_id, transition):
        self.trajectories[tls_id].append(transition)

    def clear(self):
        self.trajectories.clear()

    def set_learning_rate(self, learning_rate):
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def update(self, entropy_coef):
        samples = self._build_training_batch()
        if not samples:
            return {}

        obs = torch.as_tensor(np.stack([s["obs"] for s in samples]),
                              dtype=torch.float32, device=self.device)
        contexts = torch.as_tensor(np.stack([s["context"] for s in samples]),
                                   dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack([s["mask"] for s in samples]),
                                dtype=torch.bool, device=self.device)
        phase_maps = torch.as_tensor(np.stack([s["phase_by_action"] for s in samples]),
                                     dtype=torch.long, device=self.device)
        actions = torch.as_tensor([s["action"] for s in samples],
                                  dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([s["log_prob"] for s in samples],
                                        dtype=torch.float32, device=self.device)
        returns = torch.as_tensor([s["return"] for s in samples],
                                  dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor([s["advantage"] for s in samples],
                                     dtype=torch.float32, device=self.device)
        actor_valid = torch.as_tensor([s.get("actor_valid", True) for s in samples],
                                      dtype=torch.bool, device=self.device)
        actor_advantages = torch.zeros_like(advantages)
        valid_advantages = advantages[actor_valid]
        if valid_advantages.numel() > 1:
            actor_advantages[actor_valid] = (
                (valid_advantages - valid_advantages.mean()) /
                (valid_advantages.std(unbiased=False) + 1e-8))
        self.value_normalizer.update(returns.detach().cpu().numpy())
        normalized_returns = self.value_normalizer.normalize(returns)

        n = len(samples)
        logs = defaultdict(list)
        stop_early = False
        for _ in range(self.config.update_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.config.minibatch_size):
                idx = indices[start:start + self.config.minibatch_size]
                dist = self.network.distribution(obs[idx], masks[idx])
                new_log_probs = self._group_log_prob(
                    dist.probs, phase_maps[idx], actions[idx])
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                clipped = torch.clamp(ratio, 1.0 - self.config.ppo_clip,
                                      1.0 + self.config.ppo_clip)
                valid = actor_valid[idx]
                surrogate = torch.min(ratio * actor_advantages[idx],
                                      clipped * actor_advantages[idx])
                if torch.any(valid):
                    policy_loss = -surrogate[valid].mean()
                    entropy = self._group_entropy(
                        dist.probs, phase_maps[idx])[valid].mean()
                    approx_kl = (old_log_probs[idx][valid] -
                                 new_log_probs[valid]).mean()
                else:
                    policy_loss = new_log_probs.sum() * 0.0
                    entropy = self._group_entropy(
                        dist.probs, phase_maps[idx]).mean() * 0.0
                    approx_kl = new_log_probs.sum() * 0.0
                values = self.network.value(obs[idx], contexts[idx])
                value_loss = nn.functional.smooth_l1_loss(
                    values, normalized_returns[idx])
                loss = (policy_loss + self.config.value_coef * value_loss -
                        entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(),
                                                     self.config.max_grad_norm)
                self.optimizer.step()
                logs["policy_loss"].append(float(policy_loss.detach().cpu()))
                logs["value_loss"].append(float(value_loss.detach().cpu()))
                logs["entropy"].append(float(entropy.detach().cpu()))
                logs["approx_kl"].append(float(approx_kl.detach().cpu()))
                logs["grad_norm"].append(float(grad_norm.detach().cpu()))
                logs["clip_fraction"].append(float(((ratio - 1.0).abs() >
                                                     self.config.ppo_clip).float().mean().cpu()))
                if (self.config.target_kl > 0 and
                        float(approx_kl.detach().cpu()) > self.config.target_kl):
                    stop_early = True
                    break
            if stop_early:
                break

        explained_var = self._explained_variance(returns, obs, contexts)
        result = {key: float(np.mean(values)) for key, values in logs.items()}
        result["explained_variance"] = explained_var
        result["samples"] = n
        result["actor_samples"] = int(actor_valid.sum().item())
        result["actor_sample_fraction"] = float(actor_valid.float().mean().item())
        self.clear()
        return result

    def _build_training_batch(self):
        samples = []
        gamma = self.config.gamma_per_second
        lam = self.config.gae_lambda_per_second
        for trajectory in self.trajectories.values():
            gae = 0.0
            for transition in reversed(trajectory):
                duration = max(1, int(transition["duration"]))
                discount = gamma ** duration
                trace_discount = (gamma * lam) ** duration
                next_value = 0.0 if transition["done"] else transition["next_value"]
                delta = transition["reward"] + discount * next_value - transition["value"]
                gae = delta + (0.0 if transition["done"] else trace_discount * gae)
                item = dict(transition)
                item["advantage"] = gae
                item["return"] = gae + transition["value"]
                samples.append(item)
        return samples

    def _explained_variance(self, returns, obs, contexts):
        with torch.no_grad():
            values = self.value_normalizer.denormalize(
                self.network.value(obs, contexts))
            variance = torch.var(returns)
            if variance < 1e-8:
                return 0.0
            return float((1.0 - torch.var(returns - values) / variance).cpu())

    def _group_probabilities(self, alpha_probs, phase_maps):
        group_probs = torch.zeros_like(alpha_probs)
        group_probs.scatter_add_(1, phase_maps, alpha_probs)
        return group_probs

    def _group_log_prob(self, alpha_probs, phase_maps, actions):
        group_probs = self._group_probabilities(alpha_probs, phase_maps)
        selected_phases = phase_maps.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        selected_probs = group_probs.gather(
            1, selected_phases.unsqueeze(-1)).squeeze(-1)
        return torch.log(selected_probs.clamp_min(1e-8))

    def _group_entropy(self, alpha_probs, phase_maps):
        group_probs = self._group_probabilities(alpha_probs, phase_maps)
        return -(group_probs * torch.log(group_probs.clamp_min(1e-8))).sum(dim=-1)

    def save(self, path, metadata=None):
        torch.save({"model": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "value_normalizer": self.value_normalizer.state_dict(),
                    "obs_size": self.obs_size,
                    "alpha_bins": self.alpha_bins.tolist(),
                    "metadata": metadata or {}}, path)

    def load(self, path, load_optimizer=False):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("obs_size") != self.obs_size:
            raise ValueError("Checkpoint observation size does not match this experiment.")
        if not np.allclose(payload.get("alpha_bins", ()), self.alpha_bins):
            raise ValueError("Checkpoint alpha bins do not match this experiment.")
        self.network.load_state_dict(payload["model"])
        if "value_normalizer" in payload:
            self.value_normalizer.load_state_dict(payload["value_normalizer"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        return payload.get("metadata", {})
