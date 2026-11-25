"""
SMILE layer and model definition.

Architecture
------------
* Each **hidden** layer contains `neurons_per_type` neurons of each of the 5
  activation types (sin, mul, id, log, exp), giving 5*neurons_per_type outputs.
* The original input **x** is concatenated with the previous hidden layer's
  output at every hidden layer (jump connections).
* The **output** layer is a single identity neuron that takes the last hidden
  layer's output (plus the jump-connected original input) and produces a scalar.

Sparsity gates
--------------
Every weight w_{ij} in the linear pre-activation step is multiplied by
sigmoid(g_{ij}) where g_{ij} is a learnable gate logit.  An additional
per-neuron gate sigmoid(g_j) multiplies the full neuron output.

An L1 penalty on sigmoid(g) (both edge and activation gates) drives them
toward 0 (pruning), while the network learns larger gate logits for the edges /
neurons that matter.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from .activations import (
    safe_log_activation,
    safe_exp_activation,
    mult_activation,
)

ACTIVATION_TYPES = ["sin", "mul", "id", "log", "exp"]


class SMILELayer(nn.Module):
    """
    One hidden layer of the SMILE network.

    Parameters
    ----------
    d_in           : int  – total input dimension (original x + prev hidden)
    neurons_per_type: int – number of neurons per activation type
    """

    def __init__(self, d_in: int, neurons_per_type: int):
        super().__init__()
        self.d_in            = d_in
        self.neurons_per_type = neurons_per_type
        self.d_out           = 5 * neurons_per_type   # total output dim

        # One weight matrix per activation type  (n_neurons, d_in)
        self.W_sin  = nn.Parameter(torch.randn(neurons_per_type, d_in) * 0.1)
        self.W_mul  = nn.Parameter(torch.randn(neurons_per_type, d_in) * 0.1)
        self.W_id   = nn.Parameter(torch.randn(neurons_per_type, d_in) * 0.1)
        self.W_log  = nn.Parameter(torch.randn(neurons_per_type, d_in) * 0.1)
        self.W_exp  = nn.Parameter(torch.randn(neurons_per_type, d_in) * 0.1)

        # Edge-level gate logits (same shape as weights)
        self.G_sin  = nn.Parameter(torch.zeros(neurons_per_type, d_in))
        self.G_mul  = nn.Parameter(torch.zeros(neurons_per_type, d_in))
        self.G_id   = nn.Parameter(torch.zeros(neurons_per_type, d_in))
        self.G_log  = nn.Parameter(torch.zeros(neurons_per_type, d_in))
        self.G_exp  = nn.Parameter(torch.zeros(neurons_per_type, d_in))

        # Per-neuron activation gate logits (one per neuron per activation type)
        self.A_sin  = nn.Parameter(torch.zeros(neurons_per_type))
        self.A_mul  = nn.Parameter(torch.zeros(neurons_per_type))
        self.A_id   = nn.Parameter(torch.zeros(neurons_per_type))
        self.A_log  = nn.Parameter(torch.zeros(neurons_per_type))
        self.A_exp  = nn.Parameter(torch.zeros(neurons_per_type))

    # ── helpers ────────────────────────────────────────────────────────────────

    def _gated_weight(self, W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """Return W ⊙ σ(G): the gated (sparse) weight matrix."""
        return W * torch.sigmoid(G)

    def _act_gate(self, out: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Multiply neuron output by the per-neuron activation gate σ(A)."""
        return out * torch.sigmoid(A).unsqueeze(0)   # broadcast over batch

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, d_in)

        Returns
        -------
        h       : (batch, d_out)   concatenated outputs of all activation types
        penalty : scalar tensor    sum of range violation penalties
        """
        total_penalty = x.new_zeros(1).squeeze()

        # ── sin ────────────────────────────────────────────────────────────────
        W_sin = self._gated_weight(self.W_sin, self.G_sin)
        pre_sin = x @ W_sin.t()                         # (batch, n)
        h_sin = torch.sin(pre_sin)
        h_sin = self._act_gate(h_sin, self.A_sin)

        # ── multiplication (monomial) ──────────────────────────────────────────
        W_mul = self._gated_weight(self.W_mul, self.G_mul)
        h_mul, pen_mul = mult_activation(x, W_mul)
        h_mul = self._act_gate(h_mul, self.A_mul)
        total_penalty = total_penalty + pen_mul

        # ── identity (linear) ─────────────────────────────────────────────────
        W_id = self._gated_weight(self.W_id, self.G_id)
        pre_id = x @ W_id.t()
        h_id = pre_id
        h_id = self._act_gate(h_id, self.A_id)

        # ── log ────────────────────────────────────────────────────────────────
        W_log = self._gated_weight(self.W_log, self.G_log)
        pre_log = x @ W_log.t()                         # (batch, n)
        h_log, pen_log = safe_log_activation(pre_log)
        h_log = self._act_gate(h_log, self.A_log)
        total_penalty = total_penalty + pen_log

        # ── exp ────────────────────────────────────────────────────────────────
        W_exp = self._gated_weight(self.W_exp, self.G_exp)
        pre_exp = x @ W_exp.t()
        h_exp, pen_exp = safe_exp_activation(pre_exp)
        h_exp = self._act_gate(h_exp, self.A_exp)
        total_penalty = total_penalty + pen_exp

        h = torch.cat([h_sin, h_mul, h_id, h_log, h_exp], dim=1)
        return h, total_penalty

    # ── sparsity regularisation ────────────────────────────────────────────────

    def l1_gate_penalty(self) -> torch.Tensor:
        """
        L1 penalty on all sigmoid-gate values.
        Encourages gate values to move toward 0 (pruning).
        We actually use a bimodal penalty: gate*(1-gate) so that gates
        near 0.5 are penalised more than those near 0 or 1.
        """
        all_gates = torch.cat([
            torch.sigmoid(self.G_sin).flatten(),
            torch.sigmoid(self.G_mul).flatten(),
            torch.sigmoid(self.G_id).flatten(),
            torch.sigmoid(self.G_log).flatten(),
            torch.sigmoid(self.G_exp).flatten(),
            torch.sigmoid(self.A_sin),
            torch.sigmoid(self.A_mul),
            torch.sigmoid(self.A_id),
            torch.sigmoid(self.A_log),
            torch.sigmoid(self.A_exp),
        ])
        # Using g*(1-g) pushes gates to extremes (0 or 1);
        # L1 on the gates themselves pushes them toward 0.
        # Combined penalty: L1 + bimodal.
        return all_gates.abs().sum() + (all_gates * (1 - all_gates)).sum()


class SMILEOutputLayer(nn.Module):
    """
    Single identity-activation output neuron.

    Parameters
    ----------
    d_in : int
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.d_in = d_in
        self.W   = nn.Parameter(torch.randn(1, d_in) * 0.1)
        self.G   = nn.Parameter(torch.zeros(1, d_in))    # edge gates

    def forward(self, x: torch.Tensor):
        """x : (batch, d_in)  →  (batch, 1)"""
        W_gated = self.W * torch.sigmoid(self.G)
        return x @ W_gated.t()                            # identity activation

    def l1_gate_penalty(self) -> torch.Tensor:
        g = torch.sigmoid(self.G)
        return g.abs().sum() + (g * (1 - g)).sum()


class SMILEModel(nn.Module):
    """
    Full SMILE model.

    Parameters
    ----------
    n_inputs         : int  – number of input variables (m)
    neurons_per_type : int  – neurons per activation type per hidden layer
    n_hidden         : int or None  – number of hidden layers (default: n_inputs)
    lambda_penalty   : float – weight on the range-violation penalty
    lambda_l1        : float – weight on the sparsity gate L1 penalty
    """

    def __init__(
        self,
        n_inputs: int,
        neurons_per_type: int = 4,
        n_hidden: Optional[int] = None,
        lambda_penalty: float = 0.1,
        lambda_l1: float = 0.01,
        use_input_skip: bool = True,
    ):
        super().__init__()
        self.n_inputs        = n_inputs
        self.neurons_per_type = neurons_per_type
        self.n_hidden        = n_hidden if n_hidden is not None else n_inputs
        # Whether to concatenate the original input `x` (skip / jump connections)
        # into every hidden layer and the final output input. Default True.
        self.use_input_skip  = use_input_skip
        self.lambda_penalty  = lambda_penalty
        self.lambda_l1       = lambda_l1

        hidden_out = 5 * neurons_per_type

        self.hidden_layers = nn.ModuleList()
        for k in range(self.n_hidden):
            # First layer gets raw input; subsequent layers either get the
            # previous hidden output alone or concatenated original input +
            # previous hidden output depending on `use_input_skip`.
            if k == 0:
                d_in = n_inputs
            else:
                d_in = (n_inputs + hidden_out) if self.use_input_skip else hidden_out
            self.hidden_layers.append(SMILELayer(d_in, neurons_per_type))

        # Output layer receives jump-connected original input + last hidden
        d_out_in = (n_inputs + hidden_out) if self.use_input_skip else hidden_out
        self.output_layer = SMILEOutputLayer(d_out_in)

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, n_inputs)

        Returns
        -------
        y_pred  : (batch, 1)
        loss_extra : scalar  – combined penalty (range violations + sparsity)
        """
        total_penalty = x.new_zeros(1).squeeze()
        h = None

        for k, layer in enumerate(self.hidden_layers):
            if k == 0:
                layer_in = x
            else:
                layer_in = torch.cat([x, h], dim=1) if self.use_input_skip else h
            h, pen = layer(layer_in)
            total_penalty = total_penalty + pen

        # Final hidden → output
        out_in = torch.cat([x, h], dim=1) if self.use_input_skip else h
        y_pred = self.output_layer(out_in)

        # Sparsity regularisation
        l1 = sum(l.l1_gate_penalty() for l in self.hidden_layers)
        l1 = l1 + self.output_layer.l1_gate_penalty()

        loss_extra = (
            self.lambda_penalty * total_penalty
            + self.lambda_l1    * l1
        )
        return y_pred, loss_extra

    # ── convenience ───────────────────────────────────────────────────────────

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without computing the penalty."""
        with torch.no_grad():
            y, _ = self(x)
        return y

    def init_mult_weights(self, method: str = "uniform") -> None:
        """
        Re-initialize W_mul in every hidden layer and open the mult gates
        (G_mul, A_mul) so the neuron starts active and the initialization
        isn't immediately killed by the L1 sparsity penalty.

        Parameters
        ----------
        method : str
            "uniform"   – all weights from Uniform[-2, 2]
            "symmetric" – first half of neurons get Uniform[0.5, 2] (positive
                          exponents), second half get Uniform[-2, -0.5]
                          (negative exponents). Odd neuron count: extra neuron
                          goes to the positive half.
        """
        with torch.no_grad():
            for layer in self.hidden_layers:
                if method == "uniform":
                    layer.W_mul.uniform_(-2.0, 2.0)
                elif method == "symmetric":
                    n = layer.W_mul.shape[0]
                    n_pos = (n + 1) // 2
                    n_neg = n - n_pos
                    layer.W_mul[:n_pos].uniform_(0.5, 2.0)
                    if n_neg > 0:
                        layer.W_mul[n_pos:].uniform_(-2.0, -0.5)
                else:
                    raise ValueError(f"Unknown init_mult_weights method: {method!r}")
                # Open mult gates so the initialization actually takes effect.
                # G_mul: edge gates → +2  (sigmoid ≈ 0.88, edges start open)
                # A_mul: activation gate → +2  (neuron output starts active)
                layer.G_mul.fill_(2.0)
                layer.A_mul.fill_(2.0)

    def count_active(self) -> dict:
        """
        Return statistics on which gates are currently 'open' (σ > 0.5).
        """
        stats = {}
        for k, layer in enumerate(self.hidden_layers):
            act_open = {}
            for act in ACTIVATION_TYPES:
                A = getattr(layer, f"A_{act}")
                G = getattr(layer, f"G_{act}")
                act_open[act] = {
                    "neurons": int((torch.sigmoid(A) > 0.5).sum()),
                    "edges"  : int((torch.sigmoid(G) > 0.5).sum()),
                }
            stats[f"layer_{k}"] = act_open
        return stats
