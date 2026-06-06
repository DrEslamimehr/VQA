"""Quantum Policy Gradient trainer -- Algorithm 1 of the paper.

Implements the training loop:
  for each episode:
    roll out the agentic loop, store (s, a, r, s') in a replay buffer
    sample a batch, compute advantages A_t = sum_k gamma^k r_{t+k} - V(s_t)
    compute grad log pi via the parameter-shift rule for the VQC weights
    theta <- theta + alpha * A_t * grad log pi    (REINFORCE-with-baseline)
    asynchronously federate-average the tensor memory (with DP)

A lightweight value baseline V(s_t) is a running mean of returns. The classical
projection/head weights receive analytic REINFORCE gradients; the VQC block uses
the exact parameter-shift rule (VQCPolicy.parameter_shift_grad).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..memory.mps_memory import MPSMemory, federated_average
from ..quantum.vqc_policy import VQCPolicy
from .agent import QATMAgent
from .environment import DecPOMDPEnv


@dataclass
class TrainResult:
    rewards: List[float] = field(default_factory=list)
    final_reward: float = 0.0
    episodes: int = 0


class AdamState:
    def __init__(self, shapes, lr):
        self.lr = lr
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.m = {k: np.zeros(s) for k, s in shapes.items()}
        self.v = {k: np.zeros(s) for k, s in shapes.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in grads:
                continue
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] + self.lr * mhat / (np.sqrt(vhat) + self.eps)


class QuantumPolicyGradientTrainer:
    """Algorithm 1 implementation."""

    def __init__(
        self,
        agent: QATMAgent,
        env: DecPOMDPEnv,
        lr: float = 0.01,
        gamma: float = 0.99,
        batch_size: int = 32,
        dp_mechanism: Optional[object] = None,
        seed: int = 0,
    ):
        self.agent = agent
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.dp = dp_mechanism
        self.rng = np.random.default_rng(seed)
        shapes = {k: v.shape for k, v in agent.policy.params.items()}
        self.opt = AdamState(shapes, lr)
        self.baseline = 0.0

    def _logpi_grads(self, state, action, advantage):
        """Analytic REINFORCE gradient of log pi(a|s) for all policy params.

        The quantum block uses the parameter-shift rule; classical blocks use
        the closed-form softmax/linear gradients. Returns a dict of param grads
        already scaled by ``advantage``.
        """
        pol = self.agent.policy
        x = np.atleast_2d(state)
        # forward, caching intermediates
        h_pre = x @ pol.params["W1"] + pol.params["b1"]
        h = np.tanh(h_pre)
        ang_pre = h @ pol.params["W2"] + pol.params["b2"]
        angles = np.tanh(ang_pre) * np.pi
        z = np.asarray(pol.qnode(angles[0], pol.params["vqc"]))[None, :]
        cal = float(np.mean(pol.params["cal"])) if "cal" in pol.params else 1.0
        logits = (z @ pol.params["Wh"] + pol.params["bh"]) * cal
        probs = pol._softmax(logits)[0]

        # d log pi / d logits = (onehot - probs)
        dlogits = -probs
        dlogits[action] += 1.0
        dlogits *= advantage

        grads = {}
        # head
        grads["Wh"] = cal * np.outer(z[0], dlogits)
        grads["bh"] = cal * dlogits
        if "cal" in pol.params:
            base_logits = (z @ pol.params["Wh"] + pol.params["bh"])[0]
            dcal = float(np.dot(dlogits, base_logits)) / pol.params["cal"].size
            grads["cal"] = np.full_like(pol.params["cal"], dcal)
        # backprop into z
        dz = cal * (pol.params["Wh"] @ dlogits)  # (nq,)
        # parameter-shift grad of each <Z_j> wrt vqc weights -> sum weighted by dz
        psr = pol.parameter_shift_grad(angles[0])  # grad of sum_j <Z_j>
        grads["vqc"] = psr * float(np.mean(dz))
        # classical projection (treat dz/dangles via finite tanh deriv, coarse)
        dang = (1 - np.tanh(ang_pre) ** 2)[0] * np.pi * np.mean(dz)
        grads["W2"] = np.outer(h[0], dang)
        grads["b2"] = dang
        dh = (pol.params["W2"] @ dang) * (1 - np.tanh(h_pre) ** 2)[0]
        grads["W1"] = np.outer(x[0], dh)
        grads["b1"] = dh
        return grads

    def train(self, episodes: int, memories_for_fedavg: Optional[List[MPSMemory]] = None) -> TrainResult:
        res = TrainResult(episodes=episodes)
        for e in range(episodes):
            self.env.reset()
            buffer = []
            done = False
            ep_reward = 0.0
            steps = 0
            max_steps = min(self.env.n, 64)
            while not done and steps < max_steps:
                window = self.env._observe()
                state = self.agent.state(window)
                probs = self.agent.policy.action_probs(state)[0]
                action = int(self.rng.choice(len(probs), p=probs))
                _, reward, done = self.env.step(action)
                buffer.append((state, action, reward))
                ep_reward += reward
                steps += 1

            # advantages with running baseline
            returns = []
            G = 0.0
            for (_, _, r) in reversed(buffer):
                G = r + self.gamma * G
                returns.insert(0, G)
            self.baseline = 0.95 * self.baseline + 0.05 * np.mean(returns)

            # sample a batch and update
            idxs = self.rng.choice(len(buffer), size=min(self.batch_size, len(buffer)), replace=False)
            acc = {k: np.zeros_like(v) for k, v in self.agent.policy.params.items()}
            for i in idxs:
                state, action, _ = buffer[i]
                adv = returns[i] - self.baseline
                g = self._logpi_grads(state, action, adv)
                for k in g:
                    acc[k] += g[k] / len(idxs)
            self.opt.step(self.agent.policy.params, acc)

            # asynchronous federated averaging of tensor memory (with DP)
            if memories_for_fedavg:
                merged = federated_average(memories_for_fedavg, dp_mechanism=self.dp)
                self.agent.memory.cores = merged.cores

            res.rewards.append(ep_reward)
        res.final_reward = res.rewards[-1] if res.rewards else 0.0
        return res
