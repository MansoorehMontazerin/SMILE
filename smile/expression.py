"""
ExpressionGenerator
===================
Interprets a trained (and pruned) SMILEModel as a symbolic expression using
SymPy.

Steps
-----
1. For every active neuron (activation gate σ > threshold), collect the active
   incoming edges (edge gate σ > threshold) and their weights.
2. Build a SymPy expression for each neuron by substituting the symbolic inputs
   from earlier layers.
3. For neurons whose activation has a restricted domain (log, exp, mult), count
   the fraction of the *training* data points that fall outside the legal range.
   If > 50 %, the neuron output is replaced by the boundary constant.
4. Return the full symbolic expression for the network output.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import sympy as sp
import torch

from .model import ACTIVATION_TYPES, SMILEModel
from .activations import (
    LOG_LOWER_CUTOFF,
    EXP_UPPER_CUTOFF,
    MULT_INPUT_CUTOFF,
    MULT_INNER_CUTOFF,
)

# Defaults used when no value is passed to ExpressionGenerator.
# Override via the constructor — do not change these to configure behaviour;
# set the values in config.toml and pass them as constructor arguments.
_DEFAULT_GATE_THRESHOLD   = 0.1
_DEFAULT_WEIGHT_THRESHOLD = 1e-3
# Fraction of data points that can violate a range before we treat as constant
VIOLATION_FRACTION_LIMIT = 0.50


def _gated_numpy(W: torch.Tensor, G: torch.Tensor) -> np.ndarray:
    """Return W ⊙ σ(G) as a numpy array."""
    with torch.no_grad():
        return (W * torch.sigmoid(G)).cpu().numpy()


class ExpressionGenerator:
    """
    Convert a trained SMILEModel into a SymPy expression.

    Parameters
    ----------
    model          : SMILEModel
    X              : np.ndarray  (N, n_inputs)  – training inputs used for range checks
    var_names      : list[str] | None  – names for the input variables
                                         (default: x0, x1, …)
    threshold      : float  – sigmoid(gate) threshold to consider a neuron active
    weight_threshold : float – minimum |effective weight| to include an edge
    """

    def __init__(
        self,
        model: SMILEModel,
        X: np.ndarray,
        var_names: Optional[list] = None,
        threshold: float = _DEFAULT_GATE_THRESHOLD,
        weight_threshold: float = _DEFAULT_WEIGHT_THRESHOLD,
    ):
        self.model            = model
        self.X                = X.astype(np.float64)
        self.threshold        = threshold
        self.weight_threshold = weight_threshold
        n = model.n_inputs

        if var_names is None:
            var_names = [f"x{i}" for i in range(n)]
        self.symbols = sp.symbols(" ".join(var_names), real=True)
        if n == 1:
            self.symbols = (self.symbols,)

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self, simplify: bool = True) -> sp.Expr:
        """
        Build and return the symbolic expression for the SMILE network output.

        Parameters
        ----------
        simplify : bool  – attempt SymPy simplification (can be slow)

        Returns
        -------
        expr : sp.Expr
        """
        # Symbolic representations of the running intermediate values.
        # Replace any explicit 'const' input with the literal 1 symbol so the
        # extracted expression contains a numeric constant rather than a free
        # symbol named 'const'.
        sym_inputs: list[sp.Expr] = []
        num_inputs: list[np.ndarray] = []
        for i, s in enumerate(self.symbols):
            name = str(s)
            if name == "const":
                sym_inputs.append(sp.Integer(1))
                num_inputs.append(np.ones(len(self.X)))
            else:
                sym_inputs.append(s)
                # If X has fewer columns than symbols, fall back to ones.
                if i < self.X.shape[1]:
                    num_inputs.append(self.X[:, i])
                else:
                    num_inputs.append(np.ones(len(self.X)))

        h_sym: list[sp.Expr] = []   # current hidden layer symbolic outputs
        h_num: list[np.ndarray] = []

        for k, layer in enumerate(self.model.hidden_layers):
            if k == 0:
                ctx_sym = list(sym_inputs)
                ctx_num = list(num_inputs)
            else:
                ctx_sym = list(sym_inputs) + h_sym
                ctx_num = list(num_inputs) + h_num

            h_sym, h_num = self._process_layer(layer, ctx_sym, ctx_num, layer_idx=k)

        # Output layer
        ctx_sym = list(sym_inputs) + h_sym
        ctx_num = list(num_inputs) + h_num
        expr = self._process_output_layer(ctx_sym, ctx_num)

        if simplify:
            try:
                expr = sp.simplify(expr)
            except Exception:
                pass  # simplification is best-effort

        return expr

    # ── private helpers ───────────────────────────────────────────────────────

    def _ctx_to_array(self, ctx_num: list[np.ndarray]) -> np.ndarray:
        """Stack context numerics into (N, d) array."""
        return np.column_stack(ctx_num)

    def _active_edges(
        self,
        W_np: np.ndarray,
        G: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (indices, weights) of active edges (gate > threshold).
        W_np : (n_neurons, d_in)
        Returns arrays of length n_active.
        """
        with torch.no_grad():
            gate_vals = torch.sigmoid(G).cpu().numpy()  # (n_neurons, d_in)
        active_mask = gate_vals > self.threshold
        return active_mask, gate_vals * W_np            # effective weights

    def _neuron_active(self, A: torch.Tensor) -> np.ndarray:
        """Return bool array (n_neurons,) indicating active neurons."""
        with torch.no_grad():
            return (torch.sigmoid(A).cpu().numpy() > self.threshold)

    def _process_layer(
        self,
        layer,
        ctx_sym: list[sp.Expr],
        ctx_num: list[np.ndarray],
        layer_idx: int,
    ) -> tuple[list[sp.Expr], list[np.ndarray]]:
        """
        Compute symbolic and numeric outputs for one hidden layer.
        """
        n = layer.neurons_per_type
        X_ctx = self._ctx_to_array(ctx_num)     # (N, d_in)

        h_sym_out: list[sp.Expr]      = []
        h_num_out: list[np.ndarray]   = []

        for act in ACTIVATION_TYPES:
            W   = getattr(layer, f"W_{act}")
            G   = getattr(layer, f"G_{act}")
            A   = getattr(layer, f"A_{act}")

            W_np      = _gated_numpy(W, G)         # (n, d_in), W * σ(G)
            # Edge active when the effective weight magnitude is non-trivial.
            # This is more reliable than a raw gate threshold because the
            # optimiser may learn meaningful weights before fully saturating G.
            edge_active = np.abs(W_np) > self.weight_threshold   # (n, d_in)
            neuron_ok   = self._neuron_active(A)   # (n,)

            with torch.no_grad():
                act_gate_vals = torch.sigmoid(A).cpu().numpy()

            for j in range(n):
                if not neuron_ok[j]:
                    # Entire neuron is pruned – output 0
                    h_sym_out.append(sp.Integer(0))
                    h_num_out.append(np.zeros(len(X_ctx)))
                    continue

                # Build the pre-activation (linear combination or product)
                sym_expr, num_val = self._build_preactivation(
                    act, j, edge_active[j], W_np[j], ctx_sym, X_ctx,
                    layer_idx=layer_idx,
                )

                # Apply activation function (with range check if needed)
                sym_expr, num_val = self._apply_activation(
                    act, sym_expr, num_val
                )

                # Scale by activation gate value
                ag = float(act_gate_vals[j])
                sym_expr = sp.Rational(round(ag, 6)) * sym_expr if abs(ag - 1.0) > 1e-4 else sym_expr
                num_val  = num_val * ag

                h_sym_out.append(sym_expr)
                h_num_out.append(num_val)

        return h_sym_out, h_num_out

    def _process_output_layer(
        self,
        ctx_sym: list[sp.Expr],
        ctx_num: list[np.ndarray],
    ) -> sp.Expr:
        """Compute the symbolic output of the output (identity) layer."""
        out_layer = self.model.output_layer
        W   = out_layer.W
        G   = out_layer.G
        W_np = _gated_numpy(W, G)                   # (1, d_in)
        # Same effective-weight criterion as for hidden layers
        edge_active = np.abs(W_np[0]) > self.weight_threshold

        expr = sp.Integer(0)
        for i, (active, w_eff, s) in enumerate(
            zip(edge_active, W_np[0], ctx_sym)
        ):
            if active:
                expr = expr + sp.Float(float(w_eff)) * s

        return expr

    def _build_preactivation(
        self,
        act: str,
        j: int,
        edge_active: np.ndarray,   # (d_in,) bool
        W_eff: np.ndarray,         # (d_in,) gated weights
        ctx_sym: list[sp.Expr],
        X_ctx: np.ndarray,         # (N, d_in)
        *,
        layer_idx: int,
    ) -> tuple[sp.Expr, np.ndarray]:
        """
        Build pre-activation symbol and numeric value for neuron j of type act.
        """
        if act == "mul":
            # product activation: Π x_i^{w_i} over active edges (heuristic
            # mode only – no exact/min/max handling anymore)
            sym_expr  = sp.Integer(1)
            log_vals  = np.zeros(len(X_ctx))
            has_active = False

            # compute per-feature violation fractions for the log inputs
            feature_viol = []  # list of (index, fraction)

            for i, (active, w, s) in enumerate(
                zip(edge_active, W_eff, ctx_sym)
            ):
                if not active:
                    continue
                has_active = True

                x_i = X_ctx[:, i]
                frac_violate = np.mean(x_i < MULT_INPUT_CUTOFF)
                feature_viol.append((i, frac_violate))

                # numeric path always clips each input individually
                x_i_clipped = np.clip(x_i, MULT_INPUT_CUTOFF, None)
                log_vals += float(w) * np.log(x_i_clipped)

                # simple symbolic product; constant substitution per feature
                if frac_violate > VIOLATION_FRACTION_LIMIT:
                    const_factor = float(MULT_INPUT_CUTOFF ** float(w))
                    sym_expr = sym_expr * sp.Float(const_factor)
                else:
                    sym_expr = sym_expr * sp.Pow(s, sp.Float(float(w)))

            # heuristic: check outer exp overflow
            overflow = np.mean(log_vals > MULT_INNER_CUTOFF)
            if overflow > VIOLATION_FRACTION_LIMIT:
                const_val = float(np.exp(np.clip(log_vals, None, MULT_INNER_CUTOFF)).mean())
                viol_str = ", ".join(f"x{idx}:{fv:.2f}"
                                       for idx, fv in feature_viol if fv > 0)
                print(
                    f"  [ExprGen] layer{layer_idx} mul neuron[{j}]: "
                    f"outer exp overflow on {overflow*100:.1f}% of data "
                    f"(features {viol_str}) → constant {const_val:.4f}"
                )
                return sp.Float(const_val), np.full(len(X_ctx), const_val)

            inner_clamped = np.clip(log_vals, None, MULT_INNER_CUTOFF)
            num_val = np.exp(inner_clamped) if has_active else np.ones(len(X_ctx))
            return sym_expr, num_val        # num_val is already the final value

        else:
            # linear pre-activation
            sym_expr = sp.Integer(0)
            lin_val  = np.zeros(len(X_ctx))

            for i, (active, w, s) in enumerate(
                zip(edge_active, W_eff, ctx_sym)
            ):
                if active:
                    sym_expr = sym_expr + sp.Float(float(w)) * s
                    lin_val += float(w) * X_ctx[:, i]

            return sym_expr, lin_val

    def _apply_activation(
        self,
        act: str,
        sym_expr: sp.Expr,
        num_val: np.ndarray,
    ) -> tuple[sp.Expr, np.ndarray]:
        """
        Apply the activation to the pre-activation expression/values.
        For mul, the pre-activation is already the final value.
        """
        N = len(num_val)

        if act == "id":
            return sym_expr, num_val

        if act == "sin":
            out_num = np.sin(num_val)
            return sp.sin(sym_expr), out_num

        if act == "mul":
            # Already computed in _build_preactivation
            return sym_expr, num_val

        if act == "log":
            below = np.sum(num_val < LOG_LOWER_CUTOFF) / N
            if below > VIOLATION_FRACTION_LIMIT:
                const_val = float(np.log(LOG_LOWER_CUTOFF))
                print(
                    f"  [ExprGen] log neuron: {below*100:.1f}% of data below "
                    f"cutoff → treated as constant {const_val:.4f}"
                )
                return sp.Float(const_val), np.full(N, const_val)
            out_num  = np.log(np.clip(num_val, LOG_LOWER_CUTOFF, None))
            return sp.log(sym_expr), out_num

        if act == "exp":
            above = np.sum(num_val > EXP_UPPER_CUTOFF) / N
            if above > VIOLATION_FRACTION_LIMIT:
                const_val = float(np.exp(EXP_UPPER_CUTOFF))
                print(
                    f"  [ExprGen] exp neuron: {above*100:.1f}% of data above "
                    f"cutoff → treated as constant {const_val:.4f}"
                )
                return sp.Float(const_val), np.full(N, const_val)
            out_num = np.exp(np.clip(num_val, None, EXP_UPPER_CUTOFF))
            return sp.exp(sym_expr), out_num

        raise ValueError(f"Unknown activation type: {act}")

    # ── numeric evaluation ─────────────────────────────────────────────────────

    def evaluate_expression(
        self,
        expr: sp.Expr,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate a SymPy expression numerically on data X.

        Parameters
        ----------
        expr : sp.Expr
        X    : (N, n_inputs)

        Returns
        -------
        y : (N,) float64
        """
        f = sp.lambdify(list(self.symbols), expr, "numpy")
        args = []
        for i, s in enumerate(self.symbols):
            if str(s) == "const":
                args.append(np.ones(X.shape[0]))
            else:
                # If provided X has fewer columns than symbols, use a ones
                # column as fallback to avoid IndexError.
                if i < X.shape[1]:
                    args.append(X[:, i])
                else:
                    args.append(np.ones(X.shape[0]))
        y = np.asarray(f(*args), dtype=np.float64).ravel()
        # If expression is constant, lambdify may return a scalar (shape (1,)).
        # Broadcast it to match number of samples so boolean masks align with y.
        if y.size == 1 and X.shape[0] > 1:
            y = np.full(X.shape[0], float(y[0]), dtype=np.float64)
        return y
