"""
Parametric Optimizer and Gradient-Based Rounding
=================================================

ParametricOptimizer
-------------------
After the expression generator produces a SymPy expression (with numerical
constant coefficients), this module:
1. Replaces every numeric constant in the expression with a free parameter.
2. Fits those parameters to the training data via least-squares (scipy).
3. Re-substitutes the optimised values.

GradientRounder
---------------
For each constant in the expression it tries rounding to successive decimal
places (0 through 10).  If the R² after rounding does not drop by more than
`r2_tolerance`, the rounded value is accepted.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import torch
import copy


# ─── helpers ──────────────────────────────────────────────────────────────────

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-30)


def deep_simplify(expr):
    """
    Aggressively simplify a sympy expression:
      1. Replace Symbol('1') with 1
      2. Evaluate ALL pure-numeric sub-expressions to floats
         (handles sin(5), Rational(11,20), etc.)
      3. Rationalize floats to clean integers/fractions for algebraic cancellation
      4. Simplify (cancel, powsimp, etc.)
      5. Convert any remaining non-integer Rationals back to rounded floats
    """
    one_sym = sp.Symbol('1')
    expr = sp.expand(expr.subs(one_sym, 1))

    def _eval_numeric(e):
        if not e.free_symbols:
            try:
                val = complex(e)
                if val.imag == 0:
                    return sp.Float(val.real)
            except (TypeError, ValueError):
                pass
        if e.args:
            new_args = [_eval_numeric(a) for a in e.args]
            rebuilt = e.func(*new_args)
            if not rebuilt.free_symbols:
                try:
                    val = complex(rebuilt)
                    if val.imag == 0:
                        return sp.Float(val.real)
                except (TypeError, ValueError):
                    pass
            return rebuilt
        return e

    expr = _eval_numeric(expr)

    float_atoms = expr.atoms(sp.Float)
    if float_atoms:
        replacements = {}
        tol = 1e-3
        for f in float_atoms:
            fv = float(f)
            best = None
            for max_denom in [10, 100, 1000]:
                r = sp.Rational(f).limit_denominator(max_denom)
                if abs(float(r) - fv) < tol * max(abs(fv), 1e-10):
                    best = r
                    break
            if best is not None:
                replacements[f] = best
        if replacements:
            expr = expr.xreplace(replacements)

    expr = sp.powsimp(expr, force=True)

    candidates = [expr]
    for fn in [sp.simplify, sp.cancel, sp.radsimp, sp.powsimp]:
        try:
            candidates.append(fn(expr))
        except Exception:
            pass
    try:
        candidates.append(sp.nsimplify(expr, rational=True, tolerance=1e-3))
    except Exception:
        pass

    best = min(candidates, key=lambda e: len(str(e)))

    try:
        n, d = best.as_numer_denom()
        n_coeff, n_rest = n.as_coeff_Mul()
        if abs(n_coeff) > 1:
            d_norm = sp.expand(d / n_coeff)
            candidate = n_rest / d_norm
            rr = {}
            for atom in candidate.atoms(sp.Rational):
                if atom.q != 1:
                    fv = float(atom)
                    rounded = round(fv)
                    if abs(rounded - fv) < 0.01 * max(abs(fv), 1):
                        rr[atom] = sp.Integer(rounded)
            if rr:
                candidate = candidate.xreplace(rr)
                candidate = sp.simplify(candidate)
                if len(str(candidate)) < len(str(best)):
                    best = candidate
    except Exception:
        pass

    rat_atoms = best.atoms(sp.Rational)
    rat_replacements = {}
    for r in rat_atoms:
        if r.q != 1:
            rat_replacements[r] = sp.Float(round(float(r), 6))
    if rat_replacements:
        best = best.xreplace(rat_replacements)

    return best


def should_invert(X, y, var_names):
    """
    Determine if training on 1/y is likely easier than on y.
    Uses polynomial R² comparison: fit degree-2 poly to y and 1/y.
    """
    valid = np.isfinite(y) & (np.abs(y) > 1e-30)
    if valid.sum() < 50:
        return False

    X_v, y_v = X[valid], y[valid]
    inv_y = 1.0 / y_v

    var_cols = [i for i, v in enumerate(var_names) if v != "1"]
    if not var_cols:
        return False
    X_vars = X_v[:, var_cols]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_vars)

    r2_y_poly   = LinearRegression().fit(X_poly, y_v).score(X_poly, y_v)
    r2_inv_poly = LinearRegression().fit(X_poly, inv_y).score(X_poly, inv_y)

    if r2_inv_poly > r2_y_poly:
        print(f"    [should_invert] YES (poly R²: y={r2_y_poly:.4f}, 1/y={r2_inv_poly:.4f})")
        return True

    print(f"    [should_invert] NO (poly R²: y={r2_y_poly:.4f}, 1/y={r2_inv_poly:.4f})")
    return False


def _collect_float_atoms(expr: sp.Expr) -> list[sp.Float]:
    """Return a list of all Float leafs in the expression."""
    return [a for a in expr.atoms(sp.Float)]


# ─── ParametricOptimizer ──────────────────────────────────────────────────────

class ParametricOptimizer:
    """
    Replace numerical constants in a SymPy expression with free parameters and
    fit them to data using nonlinear least squares.

    Parameters
    ----------
    X          : (N, n_inputs) array of input data
    y          : (N,)          array of target values
    var_syms   : sequence of SymPy symbols for the input variables
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_syms: Sequence[sp.Symbol],
    ):
        self.X       = np.asarray(X, dtype=np.float64)
        self.y       = np.asarray(y, dtype=np.float64).ravel()
        self.var_syms = list(var_syms)

    def optimize(self, expr: sp.Expr) -> tuple[sp.Expr, float]:
        """
        Parametrize all Float constants in *expr* and refit them.

        Returns
        -------
        optimised_expr : sp.Expr   – expression with re-fitted constants
        r2             : float     – R² on the training data
        """
        # Collect unique float atoms
        float_atoms = sorted(
            {a for a in expr.atoms(sp.Float)},
            key=lambda a: float(a),
        )

        if not float_atoms:
            # No constants to optimise – just evaluate as-is
            y_pred = self._eval(expr)
            return expr, r2_score(self.y, y_pred)

        # Create placeholder symbols c0, c1, … for each unique float atom
        param_syms = sp.symbols(
            " ".join(f"_c{i}" for i in range(len(float_atoms)))
        )
        if len(float_atoms) == 1:
            param_syms = (param_syms,)

        # Build mapping: float atom → placeholder symbol
        atom_to_param = dict(zip(float_atoms, param_syms))

        # Substitute
        param_expr = expr.subs(list(atom_to_param.items()))

        # Initial guess from current float values
        x0 = np.array([float(a) for a in float_atoms])

        # Build lambdified function for fast evaluation
        all_syms  = self.var_syms + list(param_syms)
        try:
            f_lambda = sp.lambdify(all_syms, param_expr, "numpy")
        except Exception as e:
            warnings.warn(f"Lambdification failed: {e}; returning unoptimised.")
            y_pred = self._eval(expr)
            return expr, r2_score(self.y, y_pred)

        X = self.X
        y = self.y

        def residuals(c: np.ndarray) -> np.ndarray:
            # compute predictions for parameters c; if evaluation blows up
            # (e.g. log of negative, div-by-zero, etc.) we substitute a
            # large finite value so that least_squares can still proceed.
            args = [X[:, i] for i in range(X.shape[1])] + list(c)
            try:
                y_hat = np.asarray(f_lambda(*args), dtype=np.float64).ravel()
            except Exception:
                return np.full(len(y), 1e6)
            # clamp any non-finite entries to a big residual rather than
            # returning NaN/Inf which would trigger the "not finite" error.
            bad = ~np.isfinite(y_hat)
            if np.any(bad):
                # optional warning to help debugging
                warnings.warn(
                    f"ParametricOptimizer: {np.sum(bad)} out-of-domain "
                    "predictions replaced with large constant."
                )
                y_hat[bad] = 1e6
            return y_hat - y

        result = least_squares(
            residuals,
            x0,
            method="lm",
            max_nfev=5000,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

        # Build back-substitution map: placeholder symbol → fitted float
        param_to_fitted = {
            sym: sp.Float(round(float(val), 10))
            for sym, val in zip(param_syms, result.x)
        }

        optimised_expr = param_expr.subs(list(param_to_fitted.items()))

        y_pred = self._eval(optimised_expr)
        # drop any non-finite predictions before scoring
        finite = np.isfinite(y_pred)
        if not np.any(finite):
            warnings.warn("ParametricOptimizer: all predictions non-finite, R² set to NaN")
            return optimised_expr, float("nan")
        if not np.all(finite):
            warnings.warn(f"ParametricOptimizer: ignoring {np.sum(~finite)} non-finite points in R²")
        return optimised_expr, r2_score(self.y[finite], y_pred[finite])

    def _eval(self, expr: sp.Expr) -> np.ndarray:
        """Evaluate expression on self.X."""
        try:
            f = sp.lambdify(self.var_syms, expr, "numpy")
            args = [self.X[:, i] for i in range(self.X.shape[1])]
            y = np.asarray(f(*args), dtype=np.float64).ravel()
            if y.size == 1 and len(self.y) > 1:
                y = np.full(len(self.y), float(y[0]), dtype=np.float64)
            return y
        except Exception:
            return np.zeros(len(self.y))


# ─── GradientRounder ──────────────────────────────────────────────────────────

class GradientRounder:
    """
    Round numerical constants in a SymPy expression to the fewest decimal
    places that preserve R² above a threshold.

    Parameters
    ----------
    X           : (N, n_inputs) training data
    y           : (N,) target values
    var_syms    : list of SymPy symbols
    r2_tolerance: maximum allowed drop in R² when rounding a constant
                  (default: 0.001 → allow up to 0.1 pp drop)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        var_syms: Sequence[sp.Symbol],
        r2_tolerance: float = 0.001,
        max_decimals: int = 10,
        force_max_decimals: bool = False,
    ):
        if isinstance(max_decimals, bool) or not isinstance(max_decimals, int) or max_decimals < 0:
            raise ValueError("max_decimals must be a nonnegative integer")
        self.X            = np.asarray(X, dtype=np.float64)
        self.y            = np.asarray(y, dtype=np.float64).ravel()
        self.var_syms     = list(var_syms)
        self.r2_tolerance = r2_tolerance
        self.max_decimals = max_decimals
        self.force_max_decimals = force_max_decimals
        self.forced_count = 0

    def round_expression(self, expr: sp.Expr) -> tuple[sp.Expr, float]:
        """
        Greedily round each Float atom to the fewest permitted decimal places
        that don't decrease R² more than `r2_tolerance`.

        Returns
        -------
        rounded_expr : sp.Expr
        r2           : float
        """
        # Evaluate baseline R², ignoring any non-finite predictions
        y0 = self._eval(expr)
        finite0 = np.isfinite(y0)
        if finite0.any():
            baseline_r2 = r2_score(self.y[finite0], y0[finite0])
        else:
            baseline_r2 = float("nan")
            warnings.warn("GradientRounder: baseline predictions all non-finite")

        # Work on a mutable copy
        current_expr = expr
        self.forced_count = 0

        float_atoms = sorted(
            {a for a in current_expr.atoms(sp.Float)},
            key=lambda a: float(a),
        )

        for atom in float_atoms:
            val = float(atom)
            best_expr    = current_expr
            best_r2      = r2_score(self.y, self._eval(current_expr))
            rounded_successfully = False

            for decimals in range(self.max_decimals + 1):
                rounded_val = round(val, decimals)
                candidate = current_expr.subs(atom, sp.Float(rounded_val))
                y_cand = self._eval(candidate)
                finite_cand = np.isfinite(y_cand)
                if finite_cand.any():
                    cand_r2 = r2_score(self.y[finite_cand], y_cand[finite_cand])
                else:
                    cand_r2 = float("nan")
                if not np.isnan(cand_r2) and baseline_r2 - cand_r2 <= self.r2_tolerance:
                    best_expr  = candidate
                    best_r2    = cand_r2
                    rounded_successfully = True
                    break   # use fewest decimals that pass the threshold

            if rounded_successfully:
                current_expr = best_expr
            elif self.force_max_decimals:
                # An explicit decimal cap takes precedence over the R² budget.
                current_expr = current_expr.subs(
                    atom, sp.Float(str(round(val, self.max_decimals))))
                self.forced_count += 1

        y_fin = self._eval(current_expr)
        finite_fin = np.isfinite(y_fin)
        if finite_fin.any():
            final_r2 = r2_score(self.y[finite_fin], y_fin[finite_fin])
        else:
            final_r2 = float("nan")
            warnings.warn("GradientRounder: final predictions all non-finite")
        return current_expr, final_r2

    def _eval(self, expr: sp.Expr) -> np.ndarray:
        try:
            f = sp.lambdify(self.var_syms, expr, "numpy")
            args = [self.X[:, i] for i in range(self.X.shape[1])]
            y = np.asarray(f(*args), dtype=np.float64).ravel()
            if y.size == 1 and len(self.y) > 1:
                y = np.full(len(self.y), float(y[0]), dtype=np.float64)
            return y
        except Exception:
            return np.zeros(len(self.y))


class GreedyGatePruner:
    """
    Greedily set gate logits to a large negative value (effectively zeroing
    their sigmoid) one at a time, choosing the gate whose removal causes the
    smallest decrease in R². Stop when the best removal would reduce R² by
    more than `max_drop`.

    This operates in-place on the provided `model`.

    Parameters
    ----------
    model    : SMILEModel
    X, y     : training data (numpy arrays)
    max_drop : float  – maximum allowed R² drop for an individual removal
    min_sig  : float  – ignore gates whose sigmoid is already below this
    """

    def __init__(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        max_drop: float = 0.1,
        min_sig: float = 1e-6,
        inplace: bool = False,
    ):
        """
        Parameters
        ----------
        model: SMILEModel
            The trained model to prune. If `inplace` is False (default) the
            pruner will operate on a deep-copy so the original model is
            left untouched. The resulting pruned model is available as
            `pruner.pruned_model` after `prune()`.
        X, y: training data
        max_drop: maximum allowed R² drop for a single removal
        min_sig: ignore gates whose sigmoid is already below this
        inplace: whether to modify `model` in-place (default: False)
        """
        self.original_model = model
        # working copy: modify this during candidate evaluation and application
        self.inplace = bool(inplace)
        self._working_model = model if self.inplace else copy.deepcopy(model)

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.max_drop = float(max_drop)
        self.min_sig = float(min_sig)
        self.pruned_model = None
        self.device = next(self._working_model.parameters()).device

    def _predict_np(self) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(self.X, dtype=torch.float32, device=self.device)
            y_pred = self._working_model.predict(x_t).detach().cpu().numpy().ravel()
        return y_pred

    def _baseline_r2(self) -> float:
        y_pred = self._predict_np()
        return r2_score(self.y, y_pred)

    def _sigmoid_np(self, t: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(t).cpu().numpy()

    def _candidates(self):
        """Yield candidate descriptors for gates to try."""
        candidates = []
        for k, layer in enumerate(self._working_model.hidden_layers):
            for act in ["sin", "mul", "id", "log", "exp"]:
                G = getattr(layer, f"G_{act}")
                A = getattr(layer, f"A_{act}")
                g_sig = self._sigmoid_np(G)
                a_sig = self._sigmoid_np(A)
                # edges
                n, d = g_sig.shape
                for j in range(n):
                    for i in range(d):
                        if g_sig[j, i] > self.min_sig:
                            candidates.append(("hidden_edge", k, act, j, i))
                # activation gates (per neuron)
                for j in range(len(a_sig)):
                    if a_sig[j] > self.min_sig:
                        candidates.append(("hidden_act", k, act, j))

        # output layer edges
        G_out = self._working_model.output_layer.G  # (1, d_in)
        g_out_sig = self._sigmoid_np(G_out)[0]
        for i, s in enumerate(g_out_sig):
            if s > self.min_sig:
                candidates.append(("out_edge", i))

        return candidates

    def prune(self):
        """Perform greedy pruning. Returns (applied_list, final_r2).

        `applied_list` contains tuples describing removals.
        """
        applied = []
        baseline = self._baseline_r2()

        while True:
            candidates = self._candidates()
            if not candidates:
                break

            best = None
            best_delta = None
            best_r2 = None

            for cand in candidates:
                # Temporarily set target gate logit to large negative and evaluate
                if cand[0] == "hidden_edge":
                    _, k, act, j, i = cand
                    param = getattr(self._working_model.hidden_layers[k], f"G_{act}")
                    orig = param.data[j, i].item()
                    with torch.no_grad():
                        param.data[j, i] = -50.0
                    r2_new = r2_score(self.y, self._predict_np())
                    with torch.no_grad():
                        param.data[j, i] = orig

                elif cand[0] == "hidden_act":
                    _, k, act, j = cand
                    param = getattr(self._working_model.hidden_layers[k], f"A_{act}")
                    orig = param.data[j].item()
                    with torch.no_grad():
                        param.data[j] = -50.0
                    r2_new = r2_score(self.y, self._predict_np())
                    with torch.no_grad():
                        param.data[j] = orig

                else:  # out_edge
                    _, i = cand
                    param = self._working_model.output_layer.G
                    orig = param.data[0, i].item()
                    with torch.no_grad():
                        param.data[0, i] = -50.0
                    r2_new = r2_score(self.y, self._predict_np())
                    with torch.no_grad():
                        param.data[0, i] = orig

                delta = baseline - r2_new
                if best is None or delta < best_delta:
                    best = cand
                    best_delta = delta
                    best_r2 = r2_new

            if best is None:
                break

            # If the best removal reduces R2 by more than max_drop, stop
            if best_delta is None or best_delta > self.max_drop:
                break

            # Apply the best removal permanently to the working model
            if best[0] == "hidden_edge":
                _, k, act, j, i = best
                param = getattr(self._working_model.hidden_layers[k], f"G_{act}")
                with torch.no_grad():
                    param.data[j, i] = -50.0
                applied.append(("hidden_edge", k, act, j, i, best_delta))

            elif best[0] == "hidden_act":
                _, k, act, j = best
                param = getattr(self._working_model.hidden_layers[k], f"A_{act}")
                with torch.no_grad():
                    param.data[j] = -50.0
                applied.append(("hidden_act", k, act, j, best_delta))

            else:
                _, i = best
                param = self._working_model.output_layer.G
                with torch.no_grad():
                    param.data[0, i] = -50.0
                applied.append(("out_edge", i, best_delta))

            # Update baseline to the new R2 after applying
            baseline = best_r2

        final_r2 = baseline

        # If not inplace, expose the pruned model without modifying the original
        if not self.inplace:
            self.pruned_model = self._working_model
        else:
            self.pruned_model = self.original_model

        return applied, final_r2
