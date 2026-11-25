"""
DataGenerator
=============
Generates synthetic regression data from a symbolic expression string.

Usage
-----
    gen = DataGenerator("x0 * (1 + x1) * x2", n_inputs=3, N=20000)
    X, y = gen.generate()

Variable names default to x0, x1, ..., xm.
All variables are sampled uniformly from [low, high] (default [0, 2]).
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import sympy as sp


class DataGenerator:
    """
    Parameters
    ----------
    expression : str
        A SymPy-parseable expression string, e.g. "x0*(1+x1)*x2".
        Variable names must be x0, x1, …, x{n_inputs-1}  OR  the names
        passed in `var_names`.
    n_inputs   : int
        Number of input variables.
    N          : int
        Number of data points to generate (default 20 000).
    low, high  : float
        Sampling range for each variable (default [0, 2]).
    seed       : int | None
        Random seed for reproducibility.
    var_names  : list[str] | None
        Custom variable names (default: x0, x1, …).
    """

    def __init__(
        self,
        expression: str,
        n_inputs: int,
        N: int = 20_000,
        low: float = 0.0,
        high: float = 2.0,
        seed: Optional[int] = 42,
        var_names: Optional[list] = None,
        append_const: bool = True,
    ):
        self.expression = expression
        self.n_inputs   = n_inputs
        self.N          = N
        self.low        = low
        self.high       = high
        self.seed       = seed

        if var_names is None:
            var_names = [f"x{i}" for i in range(n_inputs)]
        # If requested, add a 'const' input name unless already present
        if append_const and "const" not in var_names:
            var_names = var_names + ["const"]
        self.var_names = var_names
        self.append_const = append_const

        # Parse expression into SymPy
        local_dict = {name: sp.Symbol(name, real=True) for name in var_names}
        self.sym_expr   = sp.sympify(expression, locals=local_dict)
        self.sym_vars   = [sp.Symbol(v, real=True) for v in var_names]

    # ── public API ─────────────────────────────────────────────────────────────

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample N data points and evaluate the expression.

        Returns
        -------
        X : (N, n_inputs)  float64 array of inputs
        y : (N,)           float64 array of outputs
        """
        rng = np.random.default_rng(self.seed)
        X   = rng.uniform(self.low, self.high, size=(self.N, self.n_inputs))

        # Optionally append a constant input column (value 1) so downstream
        # models can represent a free constant term explicitly.
        if self.append_const:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Lambdify for fast batch evaluation
        f = sp.lambdify(self.sym_vars, self.sym_expr, "numpy")
        # Build argument list matching the symbolic variables order. If a
        # constant column was appended, the corresponding symbol should be
        # present in `self.sym_vars` (it will be ignored by the expression if
        # not referenced).
        args = [X[:, i] for i in range(len(self.sym_vars))]
        y    = np.asarray(f(*args), dtype=np.float64).ravel()

        # Drop any NaN / Inf that the expression might produce
        valid = np.isfinite(y)
        if not valid.all():
            n_bad = (~valid).sum()
            print(
                f"  [DataGen] Dropped {n_bad}/{self.N} data points "
                "with non-finite outputs."
            )
            X, y = X[valid], y[valid]

        return X, y

    @property
    def true_expression(self) -> sp.Expr:
        return self.sym_expr

    @property
    def symbols(self) -> list[sp.Symbol]:
        return self.sym_vars
