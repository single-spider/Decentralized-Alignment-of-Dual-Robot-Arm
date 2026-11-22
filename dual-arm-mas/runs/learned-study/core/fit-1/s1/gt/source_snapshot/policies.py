"""Pure execution actors: one local observation and explicit protocol payloads.

Public knowledge: roles, sign convention, action bounds, candidate fractions,
tie-breaking and development-set priors. No policy closes over a world/env.
"""
import itertools
import numpy as np
from .episode import SCALE

FRACTIONS = np.array(list(itertools.product([0., .25, .5, .75, 1.], repeat=3)))


def split(error, h_a, h_b):
    correction = np.clip(np.asarray(error) / SCALE, -1, 1)
    a = -FRACTIONS * correction
    b = (1 - FRACTIONS) * correction
    cost = np.einsum("bi,ij,bj->b", a, h_a, a) + np.einsum("bi,ij,bj->b", b, h_b, b)
    best = int(np.argmin(cost))  # fixed public lexicographic tie-breaking
    return a[best] * SCALE, b[best] * SCALE


def pack_capability(observation, dtype="<f4"):
    """Six upper-triangular float32 entries, exactly 24 payload bytes."""
    return np.asarray(observation["own_H"])[np.triu_indices(3)].astype(dtype).tobytes()


def unpack_capability(payload, dtype="<f4"):
    values = np.frombuffer(payload, dtype=dtype)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Malformed capability payload")
    H = np.zeros((3, 3))
    H[np.triu_indices(3)] = values
    H += np.triu(H, 1).T
    return H


class Actor:
    def __init__(self, role, method, priors, convention=None):
        self.role = role
        self.method = method
        self.priors = {n: np.asarray(H).copy() for n, H in priors.items()}
        self.convention = np.asarray(convention if convention is not None else [.5, .5, .5])

    def act(self, own, teammate_payload=None, wire_dtype="<f4"):
        if own["role"] != self.role:
            raise ValueError("Wrong local observation")
        error = own["error"]
        correction = np.clip(error / SCALE, -1, 1) * SCALE
        fraction = {"a_only": np.ones(3), "b_only": np.zeros(3), "half": np.full(3, .5),
                    "local_convention": self.convention}.get(self.method)
        if fraction is not None:
            return -fraction * correction if self.role == "A" else (1-fraction) * correction
        if self.method not in {"local_model", "capability"}:
            raise ValueError(self.method)
        teammate = "B" if self.role == "A" else "A"
        other_H = unpack_capability(teammate_payload, wire_dtype) if self.method == "capability" else self.priors[teammate]
        a, b = split(error, own["own_H"] if self.role == "A" else other_H,
                     own["own_H"] if self.role == "B" else other_H)
        return a if self.role == "A" else b


def reconstruct_teammate(role, transition):
    """Past EXECUTED contribution, not current intended command.

    Under noise this is a noisy residual, not a ground-truth label. Own measured
    motion is essential: a raw error transition is not teammate action.
    """
    from .world import wrap
    difference = np.array(transition["error_after"]) - transition["error_before"]
    difference[2] = wrap(difference[2])
    own = np.asarray(transition["own_motion"])
    return own - difference if role == "A" else difference + own


def prediction_action(own, counts, history, history_length=4):
    counts = np.array(counts, dtype=float).copy()
    levels = np.array([0., .25, .5, .75, 1.])
    # Sliding local history: prior remains a 4-observation pseudocount, while
    # recent executed teammate fractions update beliefs. No teammate inputs.
    for transition in (history[-history_length:] if history_length else []):
        error = np.array(transition["error_before"])
        c = np.clip(error / SCALE, -1, 1) * SCALE
        past = reconstruct_teammate(own["role"], transition)
        sign = 1 if own["role"] == "A" else -1
        for axis in range(3):
            if abs(c[axis]) > SCALE[axis] * .05:
                fraction = sign * past[axis] / c[axis]
                counts[axis, np.argmin(abs(levels-fraction))] += 1
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    c = np.clip(own["error"] / SCALE, -1, 1)
    own_corrections = FRACTIONS * c
    error_after_own = own["error"] / SCALE - own_corrections
    mean = probabilities @ levels
    second = probabilities @ (levels ** 2)
    # Expected normalized terminal error over the categorical marginal model;
    # ignored teammate cost is constant w.r.t this agent's candidate action.
    weights = (SCALE / np.array([.001, .001, np.deg2rad(1)])) ** 2
    expected_error = np.sum(weights * (error_after_own**2 - 2*error_after_own*c*mean + c*c*second), axis=1)
    motion = 1000 * np.einsum("bi,ij,bj->b", own_corrections, own["own_H"], own_corrections)
    best = int(np.argmin(expected_error + motion))
    return own_corrections[best] * SCALE * (-1 if own["role"] == "A" else 1)


def reconcile_intent(own, own_payload, teammate_payload, dtype="<f4"):
    """Each actor independently normalizes its own demand, one public rule."""
    proposal = np.frombuffer(own_payload, dtype=dtype).astype(float)
    teammate = np.frombuffer(teammate_payload, dtype=dtype).astype(float)
    if proposal.shape != (3,) or teammate.shape != (3,):
        raise ValueError("Malformed intention payload")
    total = abs(proposal) + abs(teammate)
    fraction = np.divide(abs(proposal), total, out=np.full(3, .5), where=total > 1e-12)
    correction = np.clip(own["error"] / SCALE, -1, 1) * SCALE
    return correction * fraction * (-1 if own["role"] == "A" else 1)


def decide(method, observations, priors, convention, histories=None, predictor_counts=None):
    """Protocol driver routes explicit bytes, never chooses decentralized actions."""
    if method == "full_info":
        a, b = split(observations["A"]["error"], observations["A"]["own_H"], observations["B"]["own_H"])
        return {"A": a, "B": b}, {"messages": 0, "bytes": 0, "rounds": 0}
    if method in {"belief_prior", "prediction"}:
        actions = {n: prediction_action(o, predictor_counts[n], histories[n] if method == "prediction" else []) for n, o in observations.items()}
        return actions, {"messages": 0, "bytes": 0, "rounds": 0}
    if method == "intent":
        # BOTH independent proposals exist before routing either message. Own
        # proposal also uses wire precision so reconciliation is complementary.
        proposals = {n: Actor(n, "local_model", priors, convention).act(o).astype("<f4").tobytes()
                     for n, o in observations.items()}
        actions = {n: reconcile_intent(o, proposals[n], proposals["B" if n == "A" else "A"])
                   for n, o in observations.items()}
        return actions, {"messages": 2, "bytes": 24, "rounds": 1,
                         "payload_hex": {n: b.hex() for n, b in proposals.items()}}
    payloads = {n: pack_capability(o) for n, o in observations.items()} if method == "capability" else {}
    actions = {n: Actor(n, method, priors, convention).act(o, payloads.get("B" if n == "A" else "A"))
               for n, o in observations.items()}
    return actions, {"messages": len(payloads), "bytes": sum(map(len, payloads.values())), "rounds": int(bool(payloads))}


class Protocol:
    """Per-episode routing state for drop/delay/precision stress experiments."""
    def __init__(self, public, seed=0, drop=0., delay=0, dtype="<f4", history_length=4):
        self.public = public
        self.rng = np.random.default_rng(seed)
        self.drop = drop
        self.delay = delay
        self.dtype = dtype
        self.history_length = history_length
        self.sent_history = []

    def actions(self, method, observations, histories):
        priors, convention = self.public["priors"], self.public["convention"]
        if method == "prediction":
            actions = {n: prediction_action(o, self.public["predictor_counts"][n], histories[n], self.history_length)
                       for n, o in observations.items()}
            return actions, {"messages": 0, "bytes": 0, "rounds": 0}
        if method not in {"capability", "intent"}:
            return decide(method, observations, priors, convention, histories, self.public.get("predictor_counts"))
        if method == "capability":
            sent = {n: pack_capability(o, self.dtype) for n, o in observations.items()}
        else:
            sent = {n: Actor(n, "local_model", priors, convention).act(o).astype(self.dtype).tobytes()
                    for n, o in observations.items()}
        self.sent_history.append(sent)
        source_index = len(self.sent_history)-1-self.delay
        source = self.sent_history[source_index] if source_index >= 0 else {}
        actions, received = {}, 0
        for n, own in observations.items():
            other = "B" if n == "A" else "A"
            payload = source.get(other) if self.rng.random() >= self.drop else None
            received += int(payload is not None)
            if payload is None:
                actions[n] = Actor(n, "local_model", priors, convention).act(own)
            elif method == "capability":
                actions[n] = Actor(n, "capability", priors, convention).act(own, payload, self.dtype)
            else:
                actions[n] = reconcile_intent(own, sent[n], payload, self.dtype)
        return actions, {"messages": 2, "bytes": sum(map(len, sent.values())), "rounds": 1,
            "received_messages": received, "delay_steps": self.delay, "wire_dtype": self.dtype,
            "payload_hex": {n: value.hex() for n, value in sent.items()}}
