"""
run_workflow
============
End-to-end SMILE workflow:

1. Generate N = 20 000 data points from the user-provided expression.
2. Fit a sparse SMILE network and report R².
3. Extract the symbolic expression (accounting for illegal ranges).
4. Simplify and print.
5. Parametric re-optimisation → new R².
6. Simplify and print.
7. Gradient-based constant rounding → final R² and expression.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import sympy as sp
import torch

from .data        import DataGenerator
from .model       import SMILEModel
from .training    import train
from .expression  import ExpressionGenerator
from .parametric  import ParametricOptimizer, GradientRounder, r2_score


# ── helpers ────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


def _compute_r2(model: SMILEModel, X: np.ndarray, y: np.ndarray) -> float:
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model.predict(X_t).numpy().ravel()
    return r2_score(y, y_pred)


def _sympy_simplify(expr: sp.Expr) -> sp.Expr:
    """Try several SymPy simplification strategies."""
    candidates = [expr]
    for fn in (sp.simplify, sp.expand, sp.factor, sp.radsimp):
        try:
            candidates.append(fn(expr))
        except Exception:
            pass
    # Pick the shortest string representation
    return min(candidates, key=lambda e: len(str(e)))


# ── main workflow ──────────────────────────────────────────────────────────────

def run_workflow(
    expression: str,
    n_inputs: int,
    var_names: Optional[list] = None,
    # Data
    N: int = 20_000,
    data_low: float = 0.0,
    data_high: float = 2.0,
    seed: int = 42,
    # Model
    neurons_per_type: int = 4,
    n_hidden: Optional[int] = None,
    lambda_penalty: float = 0.1,
    lambda_l1: float = 0.01,
    # Training
    epochs: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 300,
    hard_example_mining: bool = False,
    hard_frac: float = 1.0,
    loss_type: str = "log-mse",
    # Pruning (apply before extraction)
    prune_max_drop: Optional[float] = None,
    prune_inplace: bool = False,
    # Rounding
    r2_tolerance: float = 0.001,
    # Extraction thresholds
    gate_threshold: float = 0.1,
    weight_threshold: float = 1e-3,
    # Misc
    verbose: bool = True,
    print_every: int = 50,
) -> dict:
    """
    Run the full SMILE symbolic-regression workflow.

    Parameters
    ----------
    expression  : str   – ground-truth expression, e.g. "x0*(1+x1)*x2"
    n_inputs    : int   – number of input variables
    var_names   : list  – optional custom variable names
    ... (see module doc for remaining args)

    Returns
    -------
    results : dict with keys
        'X', 'y', 'model', 'initial_r2', 'initial_expr',
        'parametric_expr', 'parametric_r2',
        'rounded_expr',    'rounded_r2',
        'true_expr'

    Notes
    -----
    The training-related parameters `hard_example_mining` and `hard_frac`
    control the online hard-example mining behaviour: when enabled the
    network trains each epoch on only the top `hard_frac` fraction of
    examples (by the loss metric).  `loss_type` may be 'mse', 'log-mse', or
    'r2' (negative batch R²).
    """
    if var_names is None:
        var_names = [f"x{i}" for i in range(n_inputs)]

    # ── 1. Data ────────────────────────────────────────────────────────────────
    _print_section("1 · Data Generation")
    gen  = DataGenerator(expression, n_inputs, N=N,
                         low=data_low, high=data_high, seed=seed,
                         var_names=var_names)
    X, y = gen.generate()
    print(f"  True expression : {gen.true_expression}")
    print(f"  Data shape      : X={X.shape}  y={y.shape}")
    print(f"  y range         : [{y.min():.3f}, {y.max():.3f}]")

    # ── 2. Fit SMILE ──────────────────────────────────────────────────────────
    _print_section("2 · Training Sparse SMILE Network")
    model = SMILEModel(
        n_inputs         = n_inputs,
        neurons_per_type = neurons_per_type,
        n_hidden         = n_hidden,
        lambda_penalty   = lambda_penalty,
        lambda_l1        = lambda_l1,
    )
    print(f"  Architecture: {model.n_hidden} hidden layers, "
          f"{neurons_per_type} neurons/type  →  "
          f"{5*neurons_per_type} outputs/layer")

    train(
        model, X, y,
        epochs=epochs, lr=lr,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose,
        print_every=print_every,
        hard_example_mining=hard_example_mining,
        hard_frac=hard_frac,
        loss_type=loss_type,
    )

    initial_r2 = _compute_r2(model, X, y)
    print(f"\n  → SMILE network R² = {initial_r2:.6f}")

    gate_stats = model.count_active()
    total_active_neurons = sum(
        info["neurons"]
        for layer_info in gate_stats.values()
        for info in layer_info.values()
    )
    print(f"  → Active neurons (gate > 0.5): {total_active_neurons}")

    # ── optional pruning ───────────────────────────────────────────────────
    if prune_max_drop is not None:
        _print_section("2.5 · Greedy Gate Pruning")
        from .parametric import GreedyGatePruner
        pruner = GreedyGatePruner(model, X, y,
                                  max_drop=prune_max_drop,
                                  inplace=prune_inplace)
        applied, pruned_r2 = pruner.prune()
        print("  applied removals (type, ... , r2_drop):")
        for item in applied:
            print("   ", item)
        print(f"  → R² after pruning = {pruned_r2:.6f}")
        # choose pruned model for the remainder
        model = getattr(pruner, 'pruned_model', model)

    # ── 3. Extract Expression ─────────────────────────────────────────────────
    _print_section("3 · Symbolic Expression Extraction")
    eg     = ExpressionGenerator(model, X, var_names=var_names,
                                   threshold=gate_threshold,
                                   weight_threshold=weight_threshold)
    expr   = eg.generate(simplify=False)
    expr   = _sympy_simplify(expr)

    # Compute R² for the symbolic expression, ignoring NaN/inf values
    y_expr = eg.evaluate_expression(expr, X)
    finite = np.isfinite(y_expr)
    if not finite.any():
        expr_r2 = float("nan")
    else:
        expr_r2 = r2_score(y[finite], y_expr[finite])

    print(f"\n  Extracted expression:\n    {expr}")
    print(f"\n  → Expression R² = {expr_r2:.6f}")

    # ── 4. Parametric Optimisation ────────────────────────────────────────────
    _print_section("4 · Parametric Re-Optimisation")
    var_syms = gen.symbols
    opt      = ParametricOptimizer(X, y, var_syms)
    param_expr, param_r2 = opt.optimize(expr)
    param_expr = _sympy_simplify(param_expr)

    print(f"  Optimised expression:\n    {param_expr}")
    print(f"\n  → Parametric R² = {param_r2:.6f}")

    # ── 5. Gradient Rounding ──────────────────────────────────────────────────
    _print_section("5 · Gradient-Based Constant Rounding")
    rounder = GradientRounder(X, y, var_syms, r2_tolerance=r2_tolerance)
    rounded_expr, rounded_r2 = rounder.round_expression(param_expr)
    rounded_expr = _sympy_simplify(rounded_expr)

    print(f"  Rounded expression:\n    {rounded_expr}")
    print(f"\n  → Rounded R² = {rounded_r2:.6f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_section("Summary")
    print(f"  True expression       : {gen.true_expression}")
    print(f"  SMILE network R²      : {initial_r2:.6f}")
    print(f"  Extracted expression  : {expr}")
    print(f"  Expression R²         : {expr_r2:.6f}")
    print(f"  Parametric expression : {param_expr}")
    print(f"  Parametric R²         : {param_r2:.6f}")
    print(f"  Rounded expression    : {rounded_expr}")
    print(f"  Final R²              : {rounded_r2:.6f}")
    print()

    return {
        "X"               : X,
        "y"               : y,
        "model"           : model,
        "initial_r2"      : initial_r2,
        "initial_expr"    : expr,
        "initial_expr_r2" : expr_r2,
        "parametric_expr" : param_expr,
        "parametric_r2"   : param_r2,
        "rounded_expr"    : rounded_expr,
        "rounded_r2"      : rounded_r2,
        "true_expr"       : gen.true_expression,
        "var_syms"        : var_syms,
    }
