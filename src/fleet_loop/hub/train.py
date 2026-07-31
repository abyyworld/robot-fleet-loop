"""Behaviour cloning on the assembled fleet data, in numpy.

Deliberately small and dependency-free. The claim this repo makes is about the
loop — triage, sync, validation, retraining, canary, rollback — and a training
step that needs a GPU and forty minutes would make the loop undemonstrable while
adding nothing to the claim. Swapping this for a real trainer means returning an
:class:`edge_runtime.policy.Policy`; nothing else in the hub knows what produced it.

Two details that are not incidental:

**Input normalisation is folded into the first layer after training.** The
observation columns have very different scales — joint angles, errors, a step
counter, a distance — and training on raw values wastes most of the first
layer's capacity learning the scaling. The statistics are computed from the
training set, used during training, and then folded into the weights, so the
artifact that ships is a plain MLP with no preprocessing step for the device to
get wrong. A normalisation constant that lives outside the model is a
normalisation constant that will eventually be applied twice, or not at all.

**A validation split, held out by episode boundary is not available here — the
split is by sample, and that is a real weakness.** Consecutive steps within an
episode are highly correlated, so a random sample split leaks: the validation
loss is optimistic about generalisation to *new episodes*. It is still useful
for catching divergence and for early stopping, and it is not what promotion
rests on — that is the canary's job, measured on real episodes on real nodes.
Saying so is better than reporting a number that looks like generalisation and
is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from edge_runtime.policy import ACTION_COLUMNS, Policy


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (128, 128)
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 2e-3
    val_split: float = 0.15
    # Generous, because the loss on this task plateaus and then drops again, and
    # a policy that stops at the plateau has roughly twice the action error —
    # which is the difference between 100% and 10% closed-loop success. Offline
    # loss and closed-loop success are not linearly related, and stopping early
    # on the former is a cheap way to ship the latter.
    patience: int = 30
    seed: int = 0


@dataclass
class TrainReport:
    n_samples: int
    n_train: int
    n_val: int
    epochs_run: int
    train_loss: float
    val_loss: float
    val_joint_mae_rad: float
    stopped_early: bool

    def summary(self) -> str:
        return (
            f"{self.n_samples} samples, {self.epochs_run} epochs, "
            f"val loss {self.val_loss:.5f}, val joint MAE {self.val_joint_mae_rad:.4f} rad"
            + (" (early stop)" if self.stopped_early else "")
        )

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "epochs_run": self.epochs_run,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "val_joint_mae_rad": round(self.val_joint_mae_rad, 6),
            "stopped_early": self.stopped_early,
        }


class _Adam:
    def __init__(self, params: list[np.ndarray], lr: float) -> None:
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


def train(
    observations: np.ndarray,
    actions: np.ndarray,
    obs_columns: list[str],
    *,
    config: TrainConfig | None = None,
    name: str = "fleet-bc",
) -> tuple[Policy, TrainReport]:
    """Fit a policy to (observation, action) pairs. Returns it and a report."""
    config = config or TrainConfig()
    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(actions, dtype=np.float32)
    if len(x) != len(y):
        raise ValueError(f"{len(x)} observations but {len(y)} actions")
    if len(x) < 100:
        raise ValueError(f"{len(x)} samples is not enough to train on")

    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]

    n_val = max(int(len(x) * config.val_split), 1)
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]

    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma > 1e-6, sigma, 1.0).astype(np.float32)
    zx_train = (x_train - mu) / sigma
    zx_val = (x_val - mu) / sigma

    dims = [x.shape[1], *config.hidden, y.shape[1]]
    weights = [
        (rng.normal(0, np.sqrt(2.0 / a), (a, b))).astype(np.float32)
        for a, b in zip(dims[:-1], dims[1:], strict=True)
    ]
    biases = [np.zeros(b, dtype=np.float32) for b in dims[1:]]
    params = [*weights, *biases]
    optimiser = _Adam(params, config.learning_rate)

    def forward(batch: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [batch]
        h = batch
        last = len(weights) - 1
        for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
            h = h @ w + b
            if i != last:
                h = np.tanh(h)
            activations.append(h)
        return h, activations

    best_val = np.inf
    best_state: list[np.ndarray] | None = None
    since_improved = 0
    epochs_run = 0
    train_loss = np.inf

    for _ in range(config.epochs):
        epochs_run += 1
        shuffle = rng.permutation(len(zx_train))
        losses = []
        for start in range(0, len(shuffle), config.batch_size):
            idx = shuffle[start : start + config.batch_size]
            batch_x, batch_y = zx_train[idx], y_train[idx]

            prediction, activations = forward(batch_x)
            error = prediction - batch_y
            losses.append(float((error**2).mean()))

            grad = (2.0 / len(idx)) * error
            grad_w: list[np.ndarray] = [None] * len(weights)  # type: ignore[list-item]
            grad_b: list[np.ndarray] = [None] * len(biases)  # type: ignore[list-item]
            for i in reversed(range(len(weights))):
                grad_w[i] = activations[i].T @ grad
                grad_b[i] = grad.sum(axis=0)
                if i > 0:
                    grad = (grad @ weights[i].T) * (1 - activations[i] ** 2)
            optimiser.step(params, [*grad_w, *grad_b])

        train_loss = float(np.mean(losses))
        val_prediction, _ = forward(zx_val)
        val_loss = float(((val_prediction - y_val) ** 2).mean())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = [p.copy() for p in params]
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= config.patience:
                break

    stopped_early = epochs_run < config.epochs
    if best_state is not None:
        for p, saved in zip(params, best_state, strict=True):
            p[...] = saved

    val_prediction, _ = forward(zx_val)
    val_loss = float(((val_prediction - y_val) ** 2).mean())
    val_joint_mae = float(np.abs(val_prediction[:, :7] - y_val[:, :7]).mean())

    # Fold normalisation into the first layer, so what ships is a plain MLP.
    folded_w0 = (weights[0] / sigma[:, None]).astype(np.float32)
    folded_b0 = (biases[0] - (mu / sigma) @ weights[0]).astype(np.float32)

    policy = Policy(
        [folded_w0, *weights[1:]],
        [folded_b0, *biases[1:]],
        list(obs_columns),
        list(ACTION_COLUMNS),
        name=name,
    )
    report = TrainReport(
        n_samples=len(x),
        n_train=len(x_train),
        n_val=len(x_val),
        epochs_run=epochs_run,
        train_loss=train_loss,
        val_loss=val_loss,
        val_joint_mae_rad=val_joint_mae,
        stopped_early=stopped_early,
    )
    return policy, report
