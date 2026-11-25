"""
Training utilities for the SMILE model.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from .model import SMILEModel
from .parametric import r2_score


def _effective_weight_stats(model: SMILEModel, threshold: float = 0.01) -> tuple[int, int, float]:
    """
    Count effective weights with |W * sigmoid(G)| > threshold.

    Returns
    -------
    active, total, active_percent
    """
    active = 0
    total = 0

    with torch.no_grad():
        for layer in model.hidden_layers:
            for act in ["sin", "mul", "id", "log", "exp"]:
                W = getattr(layer, f"W_{act}")
                G = getattr(layer, f"G_{act}")
                W_eff = W * torch.sigmoid(G)
                active += int((W_eff.abs() > threshold).sum().item())
                total += W_eff.numel()

        # output layer effective weights
        W_out_eff = model.output_layer.W * torch.sigmoid(model.output_layer.G)
        active += int((W_out_eff.abs() > threshold).sum().item())
        total += W_out_eff.numel()

    active_percent = (100.0 * active / total) if total > 0 else 0.0
    return active, total, active_percent


def train(
    model: SMILEModel,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 200,
    verbose: bool = True,
    print_every: int = 50,
    device: Optional[str] = None,
    hard_example_mining: bool = False,
    hard_frac: float = 1.0,
    keep_model_on_device: bool = False,
    # loss_type controls the training objective; supported values are
    # 'mse', 'log-mse', 'mae', and 'r2'.  r2 loss uses batch-wise R²
    # (minimise negative R²) and is mainly useful for diagnostics.
    loss_type: str = "log-mse",
) -> list[float]:
    """
    Train a SMILEModel with log‑MSE loss (log of the mean squared error)
    plus the built-in range/sparsity penalties.  The logarithm makes the loss
    less sensitive to target scale and smooths gradients when errors are small.

    Parameters
    ----------
    model      : SMILEModel
    X          : (N, n_inputs) float array of inputs
    y          : (N,) float array of targets
    epochs     : int   – maximum number of epochs
    lr         : float – learning rate
    batch_size : int   – mini-batch size
    patience   : int   – early-stopping patience (epochs without improvement)
    verbose    : bool  – print periodic loss messages
    print_every: int   – print frequency
    device     : str   – 'cpu' or 'cuda' (auto-detect if None)
    hard_example_mining : bool
        enable online hard‑example mining; the training set for each epoch
        is culled to the highest-error examples from the previous epoch.
        errors are measured using the same metric as the loss (see
        `loss_type`), except that R²-based loss is not supported for
        mining and falls back to squared error.
    hard_frac : float
        fraction of the dataset to keep when mining hard examples (0 < frac
        ≤ 1).  A value of 1.0 makes training equivalent to ordinary
        stochastic gradient descent on the full training set.
    loss_type : {'mse','log-mse','mae','r2'}
        choice of loss:
        * 'mse'     – ordinary mean squared error
        * 'log-mse' – log of mean squared error (default)
        * 'mae'     – mean absolute error
        * 'r2'      – negative R² computed on each batch (maximises R²).
        The loss is always combined with the model's penalty term.

    Returns
    -------
    loss_history : list of training loss per epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)

    N = len(X)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_loss    = float("inf")
    patience_cnt = 0
    loss_history = []
    gate_history = []   # mean σ(G) and σ(A) per epoch

    # we don't need persistent weights any more; `hard_frac` controls the
    # size of the subset drawn each epoch.  the selection is recomputed at
    # the top of every epoch based on the model's current errors.
    # validate requested loss type
    if loss_type not in ("mse", "log-mse", "mae", "r2"):
        raise ValueError(f"unsupported loss_type: {loss_type}")

    for epoch in range(1, epochs + 1):
        # choose which examples to visit this epoch
        if hard_example_mining:
            # compute per-example error according to loss_type.  R² loss
            # isn't defined per-sample, so fall back to squared error in that
            # case.
            with torch.no_grad():
                y_hat, _ = model(X_t)
                sqerr = (y_hat - y_t) ** 2
                if loss_type == "mse":
                    per_err = sqerr.cpu().numpy().ravel()
                elif loss_type == "log-mse":
                    per_err = torch.log(sqerr + 1e-8).cpu().numpy().ravel()
                elif loss_type == "mae":
                    per_err = torch.abs(y_hat - y_t).cpu().numpy().ravel()
                else:  # r2
                    per_err = sqerr.cpu().numpy().ravel()
            # determine how many examples to keep
            k = max(1, min(N, int(hard_frac * N)))
            if k >= N:
                perm = torch.randperm(N, device=device)
            else:
                topk = np.argsort(per_err)[-k:]
                idxs = np.random.permutation(topk)
                perm = torch.tensor(idxs, device=device, dtype=torch.long)
        else:
            perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx    = perm[start : start + batch_size]
            xb, yb = X_t[idx], y_t[idx]

            optimizer.zero_grad()
            y_hat, penalty = model(xb)
            mse  = torch.mean((y_hat - yb) ** 2)
            if loss_type == "mse":
                loss = mse + penalty
            elif loss_type == "log-mse":
                loss = torch.log(mse + 1e-8) + penalty
            elif loss_type == "mae":
                mae = torch.mean(torch.abs(y_hat - yb))
                loss = mae + penalty
            else:  # r2
                # batch R² = 1 - SS_res / SS_tot
                ss_tot = torch.var(yb, unbiased=False) * yb.numel()
                ss_res = torch.sum((y_hat - yb) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                loss = -r2 + penalty
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = ep_loss / n_batches
        loss_history.append(avg_loss)

        # Record gate statistics
        with torch.no_grad():
            all_edge_gates = []
            all_act_gates = []
            for layer in model.hidden_layers:
                for act in ["sin", "mul", "id", "log", "exp"]:
                    all_edge_gates.append(torch.sigmoid(getattr(layer, f"G_{act}")).flatten())
                    all_act_gates.append(torch.sigmoid(getattr(layer, f"A_{act}")))
            all_edge_gates.append(torch.sigmoid(model.output_layer.G).flatten())
            mean_edge = torch.cat(all_edge_gates).mean().item()
            mean_act = torch.cat(all_act_gates).mean().item()
            gate_history.append((mean_edge, mean_act))

        # Early stopping
        if avg_loss < best_loss - 1e-8:
            best_loss    = avg_loss
            patience_cnt = 0
        else:
            patience_cnt += 1

        if verbose and epoch % print_every == 0:
            # compute R² on full dataset for diagnostic purposes
            with torch.no_grad():
                y_pred = model(X_t)[0].cpu().numpy().ravel()
            r2 = r2_score(y, y_pred)
            active, total, active_pct = _effective_weight_stats(model, threshold=0.01)
            print(
                f"  Epoch {epoch:5d} / {epochs}  loss = {avg_loss:.6f}  R² = {r2:.6f}  "
                f"sparsity {active_pct:.2f}%"
            )

        if patience_cnt >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={patience}).")
            break

    model.eval()
    if not keep_model_on_device:
        model.to("cpu")
    return loss_history, gate_history
