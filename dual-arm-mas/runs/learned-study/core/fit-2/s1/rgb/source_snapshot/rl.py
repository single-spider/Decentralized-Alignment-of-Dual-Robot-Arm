"""Categorical cooperative PPO: shared local actor, separate central critic.

Actors learn three correction fractions, NOT an analytical allocation fallback.
The public action scaffold supplies signs/units: +/- fraction * clipped error.
This deliberately restricts action geometry, as do the deterministic baselines.
"""
import numpy as np
import torch
from torch import nn
from .episode import SCALE
from .policies import Actor, pack_capability, unpack_capability, reconstruct_teammate

REGIMES = ("s1", "s2", "s3", "s4")
LEVELS = np.array([0., .25, .5, .75, 1.])
INPUT_DIM = 39


def local_features(own, regime, public, history=(), received=None, proposal=None):
    """Pure one-actor boundary. No world, global state or teammate dictionary."""
    if regime not in REGIMES:
        raise ValueError(regime)
    upper = np.triu_indices(3)
    features = [np.clip(own["error"] / SCALE, -10, 10), np.asarray(own["own_q"])/3.2,
                np.clip(np.asarray(own["own_H"])[upper]*1000, -20, 20),
                np.asarray(own["previous_own_motion"])/SCALE, [1. if own["role"] == "A" else -1.]]
    other_h, belief, own_proposal, other_proposal = np.zeros(6), np.zeros(3), np.zeros(3), np.zeros(3)
    if regime == "s2" and received is not None:
        other_h = np.clip(unpack_capability(received)[upper]*1000, -20, 20)
    if regime == "s3":
        counts = np.array(public["predictor_counts"][own["role"]], dtype=float).copy()
        for transition in history[-4:]:
            c = np.clip(np.asarray(transition["error_before"])/SCALE, -1, 1)*SCALE
            past = reconstruct_teammate(own["role"], transition)
            sign = 1 if own["role"] == "A" else -1
            for axis in range(3):
                if abs(c[axis]) > SCALE[axis]*.05:
                    counts[axis, np.argmin(abs(LEVELS-sign*past[axis]/c[axis]))] += 1
        belief = (counts/counts.sum(axis=1, keepdims=True)) @ LEVELS
    if regime == "s4":
        own_proposal = np.frombuffer(proposal, dtype="<f4").astype(float)/SCALE
        if received is not None:
            other_proposal = np.frombuffer(received, dtype="<f4").astype(float)/SCALE
    features += [other_h, belief, own_proposal, other_proposal, np.eye(4)[REGIMES.index(regime)]]
    result = np.concatenate(features).astype(np.float32)
    if result.shape != (INPUT_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite/malformed actor features")
    return result


def routed_features(observations, regime, public, histories, drop=0., rng=None, delayed=None):
    """Route serialized bytes only; each actor constructs its own features."""
    sent = {}
    if regime == "s2":
        sent = {n: pack_capability(o) for n,o in observations.items()}
    elif regime == "s4":
        # Public local proposal generator is identical to the baseline's first
        # phase. Learned actors, not this generator/router, choose final actions.
        sent = {n: Actor(n, "local_model", public["priors"], public["convention"]).act(o)
                .astype("<f4").tobytes() for n,o in observations.items()}
    source = sent if delayed is None else delayed
    features = []
    for n in ("A", "B"):
        payload = source.get("B" if n == "A" else "A")
        if rng is not None and rng.random() < drop:
            payload = None
        features.append(local_features(observations[n], regime, public, histories[n], payload, sent.get(n)))
    return np.stack(features), {"bytes": sum(map(len,sent.values())), "messages":len(sent),
        "rounds":int(bool(sent)), "payload_hex": {n:v.hex() for n,v in sent.items()}}, sent


def central_features(state):
    return np.concatenate([np.asarray(state["error"])/SCALE,
                           np.asarray(state["q"]["A"])/3.2, np.asarray(state["q"]["B"])/3.2]).astype(np.float32)


def actions_from_indices(indices, observations):
    return {n: LEVELS[np.asarray(indices[i],dtype=int)] * np.clip(observations[n]["error"]/SCALE,-1,1)*SCALE
            * (-1 if n == "A" else 1) for i,n in enumerate(("A","B"))}


class CooperativePPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(INPUT_DIM,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,15))
        self.critic = nn.Sequential(nn.Linear(17,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2)); nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.)

    def distribution(self, local):
        return torch.distributions.Categorical(logits=self.actor(local).reshape(*local.shape[:-1],3,5))

    def value(self, central):
        return self.critic(central).squeeze(-1)

    @torch.no_grad()
    def choose(self, features, deterministic=False):
        dist=self.distribution(torch.as_tensor(features,dtype=torch.float32))
        indices=dist.logits.argmax(-1) if deterministic else dist.sample()
        return indices.numpy(),dist.log_prob(indices).sum(-1).numpy()


def load_policy(path):
    saved=torch.load(path,map_location="cpu",weights_only=False)
    model=CooperativePPO();model.load_state_dict(saved["model"]);model.eval()
    return model,saved
