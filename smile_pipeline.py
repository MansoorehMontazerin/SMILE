"""
smile_pipeline.py
=================
SMILE pipeline for symbolic regression on the Feynman benchmark.

For each equation (settings in config.toml):
  0. Stage 0: a log-linear R² scan decides whether one variable should be fixed.
  1. Direct SMILE (no variable fixed):
       - for each n_hidden in n_hidden_sequence, up to max_trials trials;
       - each trial checks whether 1/y is easier to fit, trains SMILE, then prunes,
         refits the coefficients and rounds them;
       - a trial whose rounded expression reaches R² >= 0.999 is a symbolic
         solution and stops the search; a trial below R² = 0.9 moves on to the
         next n_hidden.
  2. Fix-variable SMILE (a variable is fixed):
       - the largest window where the fixed variable is effectively constant is
         detected automatically;
       - stage 1 learns f(other variables) inside that window, stage 2 learns
         g(fixed variable) from the residual (y/f or y-f), and the two are combined;
       - up to max_trials trials, with the same R² >= 0.999 stopping rule.
  Training uses n_train = base_frac x (number of rows). With noise_level > 0, the
  final R² is computed against the noise-free targets.

Set eqn = "all" to run every equation. Each equation runs in its own subprocess;
results are saved to results/<eqn>.csv with a summary in results/_summary.csv, and
equations that already have a results file are skipped (resume).
"""

import sys
import time
import signal
import traceback
from pathlib import Path

# ============================================================================
# 1. IMPORTS & CONFIG
# ============================================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import numpy as np
import torch
import sympy as sp
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import DataLoader

from smile.model import SMILEModel
from smile.training import train
from smile.expression import ExpressionGenerator
from smile.parametric import ParametricOptimizer, GradientRounder, r2_score, GreedyGatePruner, deep_simplify, should_invert
from preprocess import custom_data

# ── Load config ──────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    CONFIG_PATH = Path(sys.argv[1]).resolve()
else:
    CONFIG_PATH = ROOT / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    cfg = tomllib.load(f)

feynman_cfg  = cfg["feynman"]
data_cfg     = cfg["data"]
model_cfg    = cfg["model"]
train_cfg    = cfg["training"]
post_cfg     = cfg["postprocessing"]
extract_cfg  = cfg.get("extraction", {})
pipeline_cfg = cfg.get("pipeline", {})
fixvar_cfg   = cfg.get("fixvar", {})

EQN_SETTING      = feynman_cfg["eqn"]
# Relative data paths are resolved next to the config file.
DATA_DIR         = str((CONFIG_PATH.parent / feynman_cfg["data_dir"]).resolve()) + "/"  # custom_data appends the name
FEYNMAN_META     = str((CONFIG_PATH.parent / feynman_cfg["feynman_meta"]).resolve())
BASE_FRAC        = data_cfg.get("base_frac", 0.01)
NOISE_LEVEL      = data_cfg.get("noise_level", 0)
POSITIVE_ONLY    = data_cfg.get("positive_only", False)
SEED             = data_cfg.get("seed", 42)
NEURONS_PER_TYPE = model_cfg.get("neurons_per_type", 1)
LAMBDA_PENALTY   = model_cfg.get("lambda_penalty", 1)
LAMBDA_L1        = model_cfg.get("lambda_l1", 0.05)
INIT_MULT_WEIGHTS = model_cfg.get("init_mult_weights", False)
EPOCHS           = train_cfg.get("epochs", 1000)
LR               = train_cfg.get("lr", 0.1)
BATCH_SIZE       = train_cfg.get("batch_size", 512)
PATIENCE         = train_cfg.get("patience", 300)
PRINT_EVERY      = train_cfg.get("print_every", 50)
R2_TOLERANCE     = post_cfg.get("r2_tolerance", 0.001)
WEIGHT_THRESHOLD = extract_cfg.get("weight_threshold", 1e-3)
GATE_THRESHOLD   = extract_cfg.get("gate_threshold", 0.1)
MAX_TRIALS       = pipeline_cfg.get("max_trials", 3)
STEP_TIMEOUT     = pipeline_cfg.get("step_timeout", 120)
ENABLE_FIX_VAR   = pipeline_cfg.get("enable_fix_var", True)
N_HIDDEN_SEQUENCE = pipeline_cfg.get("n_hidden_sequence", [1, 2])

hard_example_mining = train_cfg.get("hard_example_mining", False)
hard_frac           = train_cfg.get("hard_frac", 1.0)
loss_type           = train_cfg.get("loss_type", "mse")

# Fix-variable auto-range settings
MAX_WINDOW_PCT   = fixvar_cfg.get("max_window_pct", 0.25)
MIN_WINDOW_PCT   = fixvar_cfg.get("min_window_pct", 0.025)
CONSTANCY_THRESH = fixvar_cfg.get("constancy_threshold", 1)
OTHER_VAR_RANGE  = fixvar_cfg.get("other_var_range", [1.8, 2.2])
SCAN_WINDOW_FRAC = fixvar_cfg.get("scan_window_frac", 0.005)

# GPU from config (default: 0)
GPU_ID = feynman_cfg.get("gpu", 0)
GPU_SUFFIX = f"_gpu{GPU_ID}" if "gpu" in feynman_cfg else ""

_NOISE_TAG = f"_noise{NOISE_LEVEL}" if NOISE_LEVEL and float(NOISE_LEVEL) > 0 else ""
CHECKPOINT_DIR = ROOT / f"checkpoints{GPU_SUFFIX}{_NOISE_TAG}"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / f"results{GPU_SUFFIX}{_NOISE_TAG}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

SYM_R2_THRESHOLD = 0.999
POST_TIMEOUT     = pipeline_cfg.get("post_timeout", 300)

# Set per-formula by run_formula_pipeline; used to recompute final R² on noise-free y.
Y_FULL_CLEAN = None
X_FULL_CLEAN = None


def _r2_on_clean(expr, var_names):
    """Recompute R² of expr against the noise-free target y on the full dataset."""
    if Y_FULL_CLEAN is None or X_FULL_CLEAN is None:
        return None
    try:
        syms = [sp.Symbol(v) for v in var_names]
        f = sp.lambdify(syms, expr, modules=["numpy"])
        X_c = X_FULL_CLEAN[:, :len(var_names)]
        y_pred = f(*[X_c[:, i] for i in range(len(var_names))])
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_pred.ndim == 0:
            y_pred = np.full(Y_FULL_CLEAN.shape, float(y_pred))
        finite = np.isfinite(y_pred)
        if not finite.any():
            return float('nan')
        return float(r2_score(Y_FULL_CLEAN[finite], y_pred[finite]))
    except Exception as _e:
        print(f"    [clean R2] eval failed: {type(_e).__name__}: {_e}")
        return float('nan')


# ── Read metadata ────────────────────────────────────────────────────────────
meta = pd.read_csv(FEYNMAN_META)

# ── Build list of formulas to run ────────────────────────────────────────────
if "eqn_list" in feynman_cfg:
    EQN_LIST = list(feynman_cfg["eqn_list"])
    print(f"Mode: explicit list  ({len(EQN_LIST)} formulas)")
elif EQN_SETTING.lower() == "all":
    data_dir_path = Path(DATA_DIR)
    all_files = sorted([f.name for f in data_dir_path.iterdir() if f.is_file()])
    meta_names = set(meta["Filename"].values)
    EQN_LIST = [f for f in all_files if f in meta_names]
    print(f"Mode: ALL formulas  ({len(EQN_LIST)} found)")
else:
    EQN_LIST = [EQN_SETTING]
    print(f"Mode: single formula  ({EQN_SETTING})")

# ── Optionally skip equations listed in excluded_path (one name per line) ─────
EXCLUDED_PATH = feynman_cfg.get("excluded_path", "")
if EXCLUDED_PATH:
    excluded_file = Path(EXCLUDED_PATH)
    if not excluded_file.is_absolute():
        excluded_file = ROOT / excluded_file
    if excluded_file.exists():
        with open(excluded_file) as _f:
            excluded_names = {ln.strip() for ln in _f if ln.strip()}
        before = len(EQN_LIST)
        EQN_LIST = [e for e in EQN_LIST if e not in excluded_names]
        print(f"Excluded {before - len(EQN_LIST)} formulas listed in "
              f"{excluded_file.name}  →  {len(EQN_LIST)} remaining")
    else:
        print(f"WARNING: excluded_path '{excluded_file}' not found — no exclusion applied")

print(f"Device: {DEVICE}")
print(f"Config: {CONFIG_PATH}")

ACTIVATION_TYPES = ["sin", "mul", "id", "log", "exp"]


# ============================================================================
# TIMEOUT UTILITY
# ============================================================================

class StepTimeout(Exception):
    pass


class _timeout_after:
    def __init__(self, seconds, label=""):
        self.seconds = int(seconds)
        self.label = label
    def _handler(self, signum, frame):
        raise StepTimeout(f"{self.label} exceeded {self.seconds}s limit")
    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self
    def __exit__(self, *args):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old_handler)


# ============================================================================
# SYMBOLIC SOLUTION CHECK
# ============================================================================

def _clean_expr(expr):
    one_sym = sp.Symbol('1')
    expr = expr.subs(one_sym, 1)
    float_atoms = expr.atoms(sp.Float)
    if float_atoms:
        expr = expr.xreplace({f: sp.Rational(f).limit_denominator(10**12)
                               for f in float_atoms})
    sym_map = {s: sp.Symbol(s.name) for s in expr.free_symbols}
    if sym_map:
        expr = expr.subs(sym_map)
    return expr


def _is_const(expr):
    return len(expr.free_symbols) == 0


def _try_simplify_to_const(expr):
    for fn in [sp.simplify, sp.cancel, sp.expand, sp.powsimp,
               sp.trigsimp, sp.radsimp, sp.nsimplify]:
        try:
            result = fn(expr)
            if _is_const(result):
                return True, result
        except Exception:
            pass
    return False, expr


def _numerical_check_constant(expr, n_points=20):
    variables = list(expr.free_symbols)
    if not variables:
        return True
    f = sp.lambdify(variables, expr, "numpy")
    rng = np.random.default_rng(12345)
    vals = []
    for _ in range(n_points):
        point = rng.uniform(1.1, 5.0, size=len(variables))
        try:
            v = float(f(*point))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if len(vals) < 5:
        return False
    vals = np.array(vals)
    spread = np.max(np.abs(vals)) - np.min(np.abs(vals))
    scale = max(np.mean(np.abs(vals)), 1e-10)
    return spread / scale < 1e-6


def check_symbolic_solution(phi_hat, phi_true, rel_tol=1e-4):
    """
    True iff phi_hat equals phi_true up to an additive or multiplicative
    constant. Tries symbolic simplification first, then falls back to
    evaluating phi_hat and phi_true *separately* at random points and
    checking if their numeric difference or ratio is (approximately)
    constant.
    """
    phi_true = _clean_expr(phi_true)
    phi_hat  = _clean_expr(phi_hat)
    if _is_const(phi_hat):
        return False

    # ── Symbolic fast path ────────────────────────────────────────────
    try:
        diff_sym = sp.simplify(phi_true - phi_hat)
        if _is_const(diff_sym):
            return True
    except Exception:
        pass
    try:
        ratio_sym = sp.simplify(phi_true / phi_hat)
        if _is_const(ratio_sym) and ratio_sym != 0:
            return True
    except Exception:
        pass

    # ── Numerical fallback: evaluate phi_hat and phi_true separately ─
    all_syms = sorted(phi_true.free_symbols | phi_hat.free_symbols,
                      key=lambda s: s.name)
    if not all_syms:
        return False
    try:
        f_hat  = sp.lambdify(all_syms, phi_hat,  "numpy")
        f_true = sp.lambdify(all_syms, phi_true, "numpy")
    except Exception:
        return False

    rng = np.random.default_rng(12345)
    hat_vals, true_vals = [], []
    for _ in range(30):
        pt = rng.uniform(1.1, 5.0, size=len(all_syms))
        try:
            h = float(np.asarray(f_hat (*pt)).ravel()[0])
            t = float(np.asarray(f_true(*pt)).ravel()[0])
        except Exception:
            continue
        if np.isfinite(h) and np.isfinite(t):
            hat_vals.append(h)
            true_vals.append(t)
    if len(hat_vals) < 5:
        return False
    hat_vals  = np.array(hat_vals)
    true_vals = np.array(true_vals)

    # Constant difference?
    diff    = true_vals - hat_vals
    scale_d = max(np.mean(np.abs(true_vals)), np.mean(np.abs(hat_vals)), 1e-12)
    if (diff.max() - diff.min()) / scale_d < rel_tol:
        return True

    # Constant ratio?
    safe = np.abs(hat_vals) > 1e-12
    if safe.sum() >= 5:
        ratio   = true_vals[safe] / hat_vals[safe]
        scale_r = max(abs(np.mean(ratio)), 1e-12)
        if (ratio.max() - ratio.min()) / scale_r < rel_tol:
            return True

    return False


# ============================================================================
# AUTO-DETECT VARIABLE TO FIX (log-linear R² scan)
# ============================================================================

def log_linear_r2_scan(X, y, var_names, scan_window_frac=0.05):
    scores = {}
    n_vars = len(var_names)

    for fix_idx in range(n_vars):
        fix_var = var_names[fix_idx]
        col = X[:, fix_idx]
        col_min, col_max = col.min(), col.max()
        col_mid = (col_min + col_max) / 2.0
        half_w = scan_window_frac * (col_max - col_min) / 2.0

        mask = (col >= col_mid - half_w) & (col <= col_mid + half_w)
        if mask.sum() < 20:
            scores[fix_var] = float('-inf')
            print(f"    {fix_var:>10s}: too few points ({mask.sum()} < 20)")
            continue

        y_filt = y[mask]
        remaining_idx = [j for j in range(n_vars) if j != fix_idx]
        X_remaining = X[mask][:, remaining_idx]

        # Use |y|, |x| to handle formulas with negative values
        valid_mask = ((np.abs(y_filt) > 1e-30)
                      & np.all(np.abs(X_remaining) > 1e-30, axis=1)
                      & np.isfinite(y_filt))
        if valid_mask.sum() < 20:
            scores[fix_var] = float('-inf')
            continue
        log_y = np.log(np.abs(y_filt[valid_mask]))
        log_X = np.log(np.abs(X_remaining[valid_mask]))

        reg = LinearRegression().fit(log_X, log_y)
        r2 = reg.score(log_X, log_y)
        scores[fix_var] = r2

        remaining_names = [var_names[j] for j in remaining_idx]
        coefs = ", ".join(f"{v}:{c:.3f}"
                          for v, c in zip(remaining_names, reg.coef_))
        print(f"    {fix_var:>10s} fixed ({mask.sum():5d} pts) → "
              f"log-linear R² = {r2:.6f}  (coefs: {coefs})")

    return scores


def select_variable_to_fix(X, y, var_names, scan_window_frac=0.05):
    """
    Returns (best_var, scores, needs_fixing).
    """
    n_vars = len(var_names)

    if n_vars <= 1:
        print(f"    Only {n_vars} variable(s) — nothing to fix.")
        return None, {}, False

    # Full log-linear R² (no variable fixed) — use |y|, |x| to handle negatives
    X_vars = X[:, :n_vars]
    valid = ((np.abs(y) > 1e-30) & np.all(np.abs(X_vars) > 1e-30, axis=1)
             & np.isfinite(y))
    full_r2 = float('-inf')
    if valid.sum() >= 20:
        log_y = np.log(np.abs(y[valid]))
        log_X = np.log(np.abs(X_vars[valid]))
        reg = LinearRegression().fit(log_X, log_y)
        full_r2 = reg.score(log_X, log_y)
    print(f"    Full-data log-linear R² (no fixing) = {full_r2:.6f}")

    scores = log_linear_r2_scan(X, y, var_names, scan_window_frac)

    if not scores or all(v == float('-inf') for v in scores.values()):
        return None, scores, False

    best_var = max(scores, key=scores.get)
    best_r2 = scores[best_var]

    improvement = abs(best_r2 - full_r2)
    rel_improvement = improvement / max(1.0 - full_r2, 1e-9)
    print(f"    Full R² = {full_r2:.6f},  Best fixed R² = {best_r2:.6f} "
          f"({best_var}),  Improvement = {improvement:.6f}  "
          f"(rel = {rel_improvement:.3f})")

    if rel_improvement < 0.1:
        print(f"    → Formula is a true monomial. No variable needs fixing.")
        return None, scores, False

    print(f"    ★ Best variable to fix: {best_var} "
          f"(log-linear R² = {best_r2:.6f})")

    return best_var, scores, True


# ============================================================================
# AUTO-DETECT FIX RANGE (regression-based constancy test)
# ============================================================================

def find_fix_range_regression(X, y, var_names, fix_var_name,
                              fix_point, max_pct, min_pct, n_windows=6,
                              other_range=None, constancy_thresh=0.05):
    """
    Try n_windows window sizes from [fix_point ± max_pct*fix_point] down to
    [fix_point ± min_pct*fix_point].  For each window:
      1. Filter data: fix_var in window AND other vars in other_range.
      2. Fit y ~ beta * x_fix  (single-variable linear regression).
      3. C = |beta| * window_width / std(y).
      4. If C < constancy_thresh, the window passes.
    Return the largest passing window.
    """
    fix_idx = var_names.index(fix_var_name)
    other_lo, other_hi = other_range or [1.95, 2.05]

    pcts = np.logspace(np.log10(max_pct), np.log10(min_pct), n_windows)

    print(f"\n    [find_fix_range_regression] fix_var={fix_var_name}, fix_point={fix_point}")
    print(f"      Testing {n_windows} windows: ±{max_pct*100:.1f}% to ±{min_pct*100:.1f}%")
    print(f"      Other vars constrained to [{other_lo}, {other_hi}]")
    print(f"      Constancy threshold: {constancy_thresh}")

    best_range = None

    for pct in pcts:
        half_w = pct * fix_point
        lo = fix_point - half_w
        hi = fix_point + half_w
        window_width = hi - lo

        # Filter: fix_var in [lo, hi]
        fix_col = X[:, fix_idx]
        mask = (fix_col >= lo) & (fix_col <= hi)

        # Filter: all other vars in [other_lo, other_hi]
        for j in range(len(var_names)):
            if j == fix_idx:
                continue
            mask = mask & (X[:, j] >= other_lo) & (X[:, j] <= other_hi)

        n_pts = mask.sum()
        if n_pts < 10:
            print(f"      ±{pct*100:5.1f}%  [{lo:.4f}, {hi:.4f}]  "
                  f"pts={n_pts:5d}  SKIP (too few)")
            continue

        x_fix = X[mask, fix_idx].reshape(-1, 1)
        y_filt = y[mask]

        std_y = np.std(y_filt)
        if std_y < 1e-30:
            print(f"      ±{pct*100:5.1f}%  [{lo:.4f}, {hi:.4f}]  "
                  f"pts={n_pts:5d}  SKIP (std(y)≈0)")
            continue

        reg = LinearRegression().fit(x_fix, y_filt)
        beta = abs(reg.coef_[0])

        C = beta * window_width / std_y

        passed = C < constancy_thresh
        status = "PASS" if passed else "FAIL"
        print(f"      ±{pct*100:5.1f}%  [{lo:.4f}, {hi:.4f}]  "
              f"pts={n_pts:5d}  C={C:.6f}  beta={beta:.6f}  {status}")

        if passed and best_range is None:
            best_range = [lo, hi]

    if best_range is None:
        half_w = min_pct * fix_point
        best_range = [fix_point - half_w, fix_point + half_w]
        print(f"      No window passed threshold — using smallest: {best_range}")
    else:
        print(f"      ★ Best (largest passing): {best_range}")

    return best_range


# ============================================================================
# HELPER: run one SMILE trial (direct, no fix-variable)
# ============================================================================

def run_direct_trial(eqn, true_formula, var_names_base,
                     X_FULL, y_FULL, n_train,
                     n_hidden, trial_idx, trial_seed):
    """
    Run a single direct SMILE trial with auto-invert detection.
    """
    t_start = time.time()

    try:
        # Subsample full data to n_train
        n_total = len(y_FULL)
        if n_total > n_train:
            rng = np.random.default_rng(trial_seed)
            idx = rng.choice(n_total, size=n_train, replace=False)
            X = X_FULL[idx].copy()
            y = y_FULL[idx].copy()
            print(f"    Subsampled to {n_train} / {n_total}")
        else:
            X = X_FULL.copy()
            y = y_FULL.copy()
        VAR_NAMES = var_names_base + ['1']
        N_INPUTS = X.shape[1]
        X_eval = X.copy()

        # Auto-detect inversion
        invert_target = should_invert(X, y, VAR_NAMES)

        y_orig = y.copy()
        if invert_target:
            nonzero = np.abs(y) > 1e-30
            if not nonzero.all():
                n_drop = (~nonzero).sum()
                print(f"    [invert] Dropping {n_drop} samples with y~0")
                X, y, y_orig = X[nonzero], y[nonzero], y_orig[nonzero]
                X_eval = X_eval[nonzero]
            y = (1.0 / y).astype(np.float32)
            print(f"    [invert] Training on 1/y")

        print(f"    n_inputs={N_INPUTS}  X={X.shape}  y=[{y.min():.3f}, {y.max():.3f}]")

        # ── TRAIN ────────────────────────────────────────────────────────
        model = SMILEModel(
            n_inputs=N_INPUTS, neurons_per_type=NEURONS_PER_TYPE,
            n_hidden=n_hidden, lambda_penalty=LAMBDA_PENALTY,
            lambda_l1=LAMBDA_L1,
        )
        if INIT_MULT_WEIGHTS:
            model.init_mult_weights(method=INIT_MULT_WEIGHTS)
        model = model.to(DEVICE)

        train(
            model, X, y, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
            patience=PATIENCE, print_every=PRINT_EVERY, verbose=True,
            device=str(DEVICE), hard_example_mining=hard_example_mining,
            hard_frac=hard_frac, loss_type=loss_type,
            keep_model_on_device=True,
        )

        ckpt = CHECKPOINT_DIR / f"direct_{eqn}_h{n_hidden}_t{trial_idx}.pt"
        torch.save(model.state_dict(), ckpt)

        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            y_net = model.predict(X_t).detach().cpu().numpy().ravel()
        network_r2 = float(r2_score(y, y_net))
        print(f"    Network R2 = {network_r2:.6f}")

        # ── PRUNE ────────────────────────────────────────────────────────
        with _timeout_after(STEP_TIMEOUT, "Pruning"):
            pruner = GreedyGatePruner(model, X, y, max_drop=0.01, inplace=False)
            applied, final_r2 = pruner.prune()
            pruned_model = pruner.pruned_model
            print(f"    Pruned R2 = {final_r2:.6f}  ({len(applied)} removals)")

        # ── EXTRACT ──────────────────────────────────────────────────────
        with _timeout_after(POST_TIMEOUT, "Extraction"):
            eg = ExpressionGenerator(
                pruned_model, X, var_names=VAR_NAMES,
                threshold=GATE_THRESHOLD, weight_threshold=WEIGHT_THRESHOLD,
            )
            expr = eg.generate(simplify=True)
            print(f"    Extracted: {expr}")

        # ── PARAMETRIC ───────────────────────────────────────────────────
        with _timeout_after(STEP_TIMEOUT, "Parametric"):
            var_syms = [sp.Symbol(v) for v in VAR_NAMES]
            opt = ParametricOptimizer(X_eval, y, var_syms)
            param_expr, _ = opt.optimize(expr)
            try:
                param_expr = sp.simplify(param_expr)
            except Exception:
                pass

        # ── ROUND ────────────────────────────────────────────────────────
        with _timeout_after(STEP_TIMEOUT, "Rounding"):
            rounder = GradientRounder(X_eval, y, var_syms, r2_tolerance=R2_TOLERANCE)
            rounded_expr, rounded_r2 = rounder.round_expression(param_expr)
            try:
                rounded_expr = sp.simplify(rounded_expr)
            except Exception:
                pass
            print(f"    Rounded: {rounded_expr}   R2={rounded_r2:.6f}")

        # ── INVERT ───────────────────────────────────────────────────────
        with _timeout_after(POST_TIMEOUT, "Inversion"):
            if invert_target:
                rounded_expr = sp.simplify(1 / rounded_expr)
                y = y_orig
                y_round = eg.evaluate_expression(rounded_expr, X_eval)
                finite = np.isfinite(y_round)
                rounded_r2 = r2_score(y[finite], y_round[finite]) if finite.any() else float('nan')
                print(f"    Inverted: {rounded_expr}   R2={rounded_r2:.6f}")

        # ── DEEP SIMPLIFY ────────────────────────────────────────────────
        with _timeout_after(POST_TIMEOUT, "DeepSimplify"):
            rounded_expr = deep_simplify(rounded_expr)
            print(f"    Simplified: {rounded_expr}")

        # ── CLEAN-DATA R² (recompute on noise-free y for sym check) ─────
        if NOISE_LEVEL and float(NOISE_LEVEL) > 0:
            clean_r2 = _r2_on_clean(rounded_expr, var_names_base)
            if clean_r2 is not None and np.isfinite(clean_r2):
                print(f"    Clean R2 = {clean_r2:.6f}  (noisy was {rounded_r2:.6f})")
                rounded_r2 = clean_r2

        # ── SYMBOLIC CHECK ───────────────────────────────────────────────
        r2_sym = bool(np.isfinite(rounded_r2) and rounded_r2 >= SYM_R2_THRESHOLD)
        is_sym = r2_sym

        elapsed = time.time() - t_start
        print(f"    R2_sym: {r2_sym}  symbolic_solution: {is_sym}  Time: {elapsed:.1f}s")

        return {
            "formula_name":       eqn,
            "ground_truth":       true_formula,
            "rounded_expression": str(rounded_expr),
            "rounded_r2":         rounded_r2,
            "network_r2":         network_r2,
            "r2_sym":             r2_sym,
            "symbolic_solution":  is_sym,
            "noise_level":        NOISE_LEVEL,
            "invert_target":      invert_target,
            "n_hidden":           n_hidden,
            "phase":              "direct",
            "time_sec":           round(elapsed, 2),
            "trial":              trial_idx,
        }

    except StepTimeout as e:
        print(f"\n    !!! TIMEOUT: {e}")
        raise

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"\n    !!! TRIAL FAILED: {type(e).__name__}: {e}")
        return {
            "formula_name":       eqn,
            "ground_truth":       true_formula,
            "rounded_expression": float('nan'),
            "rounded_r2":         float('nan'),
            "network_r2":         float('nan'),
            "r2_sym":             False,
            "symbolic_solution":  False,
            "noise_level":        NOISE_LEVEL,
            "invert_target":      False,
            "n_hidden":           n_hidden,
            "phase":              "direct",
            "time_sec":           round(elapsed, 2),
            "trial":              trial_idx,
        }


# ============================================================================
# HELPER: run one SMILE sub-pipeline (for fix-variable stages)
# ============================================================================

def _run_smile_substage(X, y, var_names, checkpoint_path, stage_label="",
                        invert_target=False):
    """Train → prune → extract → parametric → round (n_hidden=1)."""
    n_inputs = X.shape[1]

    y_orig = y.copy()
    if invert_target:
        nonzero = np.abs(y) > 1e-30
        if not nonzero.all():
            n_drop = (~nonzero).sum()
            print(f"      [invert] Dropping {n_drop} samples with y~0")
            X, y, y_orig = X[nonzero], y[nonzero], y_orig[nonzero]
        y = (1.0 / y).astype(np.float32)
        print(f"      [invert] Training on 1/y")

    print(f"      {stage_label}: n_inputs={n_inputs}  X={X.shape}  "
          f"y=[{y.min():.4f}, {y.max():.4f}]")

    model = SMILEModel(
        n_inputs=n_inputs, neurons_per_type=NEURONS_PER_TYPE,
        n_hidden=1, lambda_penalty=LAMBDA_PENALTY, lambda_l1=LAMBDA_L1,
    )
    if INIT_MULT_WEIGHTS:
        model.init_mult_weights(method=INIT_MULT_WEIGHTS)
    model = model.to(DEVICE)

    train(
        model, X, y, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
        patience=PATIENCE, print_every=PRINT_EVERY, verbose=True,
        device=str(DEVICE), hard_example_mining=hard_example_mining,
        hard_frac=hard_frac, loss_type=loss_type,
        keep_model_on_device=True,
    )
    torch.save(model.state_dict(), checkpoint_path)

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        y_net = model.predict(X_t).detach().cpu().numpy().ravel()
    network_r2 = float(r2_score(y, y_net))
    print(f"      Network R2 = {network_r2:.6f}")

    with _timeout_after(STEP_TIMEOUT, "Pruning"):
        pruner = GreedyGatePruner(model, X, y, max_drop=0.01, inplace=False)
        _, final_r2 = pruner.prune()
        pruned_model = pruner.pruned_model
        print(f"      Pruned R2 = {final_r2:.6f}")

    with _timeout_after(POST_TIMEOUT, "Extraction"):
        eg = ExpressionGenerator(
            pruned_model, X, var_names=var_names,
            threshold=GATE_THRESHOLD, weight_threshold=WEIGHT_THRESHOLD,
        )
        expr = eg.generate(simplify=True)

    with _timeout_after(STEP_TIMEOUT, "Parametric"):
        var_syms = [sp.Symbol(v) for v in var_names]
        opt = ParametricOptimizer(X, y, var_syms)
        param_expr, _ = opt.optimize(expr)
        try:
            param_expr = sp.simplify(param_expr)
        except Exception:
            pass

    with _timeout_after(STEP_TIMEOUT, "Rounding"):
        rounder = GradientRounder(X, y, var_syms, r2_tolerance=R2_TOLERANCE)
        rounded_expr, rounded_r2 = rounder.round_expression(param_expr)
        try:
            rounded_expr = sp.simplify(rounded_expr)
        except Exception:
            pass
        print(f"      Rounded: {rounded_expr}   R2={rounded_r2:.6f}")

    with _timeout_after(POST_TIMEOUT, "Inversion"):
        if invert_target:
            rounded_expr = sp.simplify(1 / rounded_expr)
            y = y_orig
            y_round = eg.evaluate_expression(rounded_expr, X)
            finite = np.isfinite(y_round)
            rounded_r2 = r2_score(y[finite], y_round[finite]) if finite.any() else float('nan')
            print(f"      Inverted: {rounded_expr}   R2={rounded_r2:.6f}")

    return {"rounded_expr": rounded_expr, "rounded_r2": rounded_r2,
            "network_r2": network_r2, "eg": eg}


# ============================================================================
# HELPER: run one fix-variable trial (real data, auto-detected range)
# ============================================================================

def run_fixvar_trial(eqn, true_formula, var_names_base,
                     X_FULL, y_FULL, n_train,
                     fix_var_name, fix_var_range, trial_idx, trial_seed):
    """
    Fix-variable trial using real data:
      Stage 1: filter by fix_var_range (auto-detected), subsample, learn f(free_vars)
      Stage 2: compute residual on full data, subsample by base_frac, learn g(fix_var)
      Stage 3: combine f * g or f + g
    """
    t_start = time.time()

    try:
        all_var_names = var_names_base
        fix_var_idx = all_var_names.index(fix_var_name)
        free_var_names = [v for v in all_var_names if v != fix_var_name]
        fix_lo, fix_hi = fix_var_range

        # ── STAGE 1: Filter & learn f(free_vars) ────────────────────────
        print(f"\n    STAGE 1: Fix {fix_var_name} ∈ [{fix_lo}, {fix_hi}]")

        fix_col = X_FULL[:, fix_var_idx]
        range_mask = (fix_col >= fix_lo) & (fix_col <= fix_hi)
        n_in_range = range_mask.sum()
        print(f"      Samples in range: {n_in_range}")

        if n_in_range < 50:
            raise ValueError(f"Only {n_in_range} samples in range — too few")

        # Subsample to n_train (= base_frac × total dataset)
        range_indices = np.where(range_mask)[0]
        if len(range_indices) > n_train:
            rng = np.random.default_rng(trial_seed)
            range_indices = rng.choice(range_indices, size=n_train, replace=False)
            print(f"      Subsampled to {n_train} / {n_in_range}")
        else:
            print(f"      Using all {len(range_indices)} samples (< n_train={n_train})")

        free_col_indices = [i for i in range(len(all_var_names)) if i != fix_var_idx]
        const_col_idx = len(all_var_names)
        X_s1 = np.hstack([
            X_FULL[range_indices][:, free_col_indices],
            X_FULL[range_indices][:, const_col_idx:const_col_idx+1],
        ]).astype(np.float32)
        y_s1 = y_FULL[range_indices].astype(np.float32)
        names_s1 = free_var_names + ["1"]

        # Auto-invert for Stage 1
        invert_s1 = should_invert(X_s1, y_s1, names_s1)

        ckpt_s1 = CHECKPOINT_DIR / f"fixvar_{eqn}_t{trial_idx}_s1.pt"
        res_s1 = _run_smile_substage(
            X_s1, y_s1, names_s1, ckpt_s1,
            stage_label=f"S1: f({','.join(free_var_names)})",
            invert_target=invert_s1,
        )
        f_expr = res_s1["rounded_expr"]
        s1_r2 = res_s1["rounded_r2"]
        print(f"      Stage 1: {f_expr}  R2={s1_r2:.6f}")

        # ── STAGE 2: Residual → g(fix_var) ──────────────────────────────
        print(f"\n    STAGE 2: Residual → g({fix_var_name})")

        # Subsample full data to n_train
        n_s2_total = len(y_FULL)
        if n_s2_total > n_train:
            rng_s2 = np.random.default_rng(trial_seed + 1000)
            s2_idx = rng_s2.choice(n_s2_total, size=n_train, replace=False)
            X_s2_full = X_FULL[s2_idx]
            y_s2_full = y_FULL[s2_idx]
            print(f"      Subsampled to {n_train} / {n_s2_total}")
        else:
            X_s2_full = X_FULL
            y_s2_full = y_FULL
            print(f"      Using all {n_s2_total} samples (< n_train={n_train})")

        # Evaluate f on the subsampled data
        free_col_indices_full = [all_var_names.index(v) for v in free_var_names]
        X_free_sub = np.hstack([
            X_s2_full[:, free_col_indices_full],
            X_s2_full[:, const_col_idx:const_col_idx+1],
        ])

        f_syms = [sp.Symbol(v) for v in free_var_names + ["1"]]
        f_func = sp.lambdify(f_syms, f_expr, "numpy")
        f_args = [X_free_sub[:, i] for i in range(X_free_sub.shape[1])]
        f_vals = np.asarray(f_func(*f_args), dtype=np.float64).ravel()
        if f_vals.size == 1:
            f_vals = np.full(len(y_s2_full), float(f_vals[0]))

        # Auto-detect decomposition
        fix_col_sub = X_s2_full[:, fix_var_idx]
        safe_f = np.where(np.abs(f_vals) > 1e-30, f_vals, np.nan)
        r_mult = y_s2_full / safe_f
        r_add = y_s2_full - f_vals
        finite_mult = np.isfinite(r_mult)
        finite_add = np.isfinite(r_add)

        def _dep(residual, mask):
            r = residual[mask]
            Xf = X_free_sub[mask]
            if len(r) < 10:
                return float('inf')
            corrs = []
            for j in range(Xf.shape[1] - 1):
                c = np.corrcoef(r, Xf[:, j])[0, 1]
                corrs.append(abs(c) if np.isfinite(c) else 1.0)
            return np.mean(corrs) if corrs else 0.0

        if _dep(r_mult, finite_mult) <= _dep(r_add, finite_add):
            decomposition = "multiplicative"
            residual, finite_mask = r_mult, finite_mult
        else:
            decomposition = "additive"
            residual, finite_mask = r_add, finite_add
        print(f"      Decomposition: {decomposition}")

        X_s2 = np.column_stack([
            fix_col_sub[finite_mask], np.ones(finite_mask.sum()),
        ]).astype(np.float32)
        y_s2 = residual[finite_mask].astype(np.float32)
        names_s2 = [fix_var_name, "1"]

        # Auto-invert for Stage 2
        invert_s2 = should_invert(X_s2, y_s2, names_s2)

        ckpt_s2 = CHECKPOINT_DIR / f"fixvar_{eqn}_t{trial_idx}_s2.pt"
        res_s2 = _run_smile_substage(
            X_s2, y_s2, names_s2, ckpt_s2,
            stage_label=f"S2: g({fix_var_name})",
            invert_target=invert_s2,
        )
        g_expr = res_s2["rounded_expr"]
        print(f"      Stage 2: {g_expr}  R2={res_s2['rounded_r2']:.6f}")

        # ── STAGE 3: Combine ─────────────────────────────────────────────
        const_sym = sp.Symbol("1")
        f_clean = f_expr.subs(const_sym, 1)
        g_clean = g_expr.subs(const_sym, 1)

        if decomposition == "multiplicative":
            combined_expr = f_clean * g_clean
        else:
            combined_expr = f_clean + g_clean
        try:
            combined_expr = sp.simplify(combined_expr)
        except Exception:
            pass

        # Parametric + rounding on combined expression using n_train samples
        all_syms = [sp.Symbol(v) for v in all_var_names]
        X_eval = X_FULL[:, :len(all_var_names)]
        if len(y_FULL) > n_train:
            rng_s3 = np.random.default_rng(trial_seed + 2000)
            s3_idx = rng_s3.choice(len(y_FULL), size=n_train, replace=False)
            X_s3, y_s3 = X_eval[s3_idx], y_FULL[s3_idx]
        else:
            X_s3, y_s3 = X_eval, y_FULL

        with _timeout_after(STEP_TIMEOUT, "Parametric-combined"):
            opt = ParametricOptimizer(X_s3, y_s3, all_syms)
            combined_expr, _ = opt.optimize(combined_expr)
            try:
                combined_expr = sp.simplify(combined_expr)
            except Exception:
                pass

        with _timeout_after(STEP_TIMEOUT, "Rounding-combined"):
            rounder = GradientRounder(X_s3, y_s3, all_syms,
                                       r2_tolerance=R2_TOLERANCE)
            combined_expr, combined_r2 = rounder.round_expression(combined_expr)
            try:
                combined_expr = sp.simplify(combined_expr)
            except Exception:
                pass
            print(f"      Combined (rounded): {combined_expr}   R2={combined_r2:.6f}")

        # Deep simplify the final combined expression
        with _timeout_after(POST_TIMEOUT, "DeepSimplify-combined"):
            combined_expr = deep_simplify(combined_expr)
            print(f"      Simplified: {combined_expr}")

        # Clean-data R² (recompute on noise-free y for sym check)
        if NOISE_LEVEL and float(NOISE_LEVEL) > 0:
            clean_r2 = _r2_on_clean(combined_expr, all_var_names)
            if clean_r2 is not None and np.isfinite(clean_r2):
                print(f"      Clean R2 = {clean_r2:.6f}  (noisy was {combined_r2:.6f})")
                combined_r2 = clean_r2

        # Symbolic check
        r2_sym = bool(np.isfinite(combined_r2) and combined_r2 >= SYM_R2_THRESHOLD)
        is_sym = r2_sym

        elapsed = time.time() - t_start

        print(f"      Combined: {combined_expr}   R2={combined_r2:.6f}")
        print(f"      R2_sym: {r2_sym}  symbolic_solution: {is_sym}  Time: {elapsed:.1f}s")

        return {
            "formula_name":       eqn,
            "ground_truth":       true_formula,
            "rounded_expression": str(combined_expr),
            "rounded_r2":         combined_r2,
            "network_r2":         res_s1.get("network_r2", float('nan')),
            "r2_sym":             r2_sym,
            "symbolic_solution":  is_sym,
            "noise_level":        NOISE_LEVEL,
            "invert_target":      f"fixvar(s1={invert_s1},s2={invert_s2})",
            "n_hidden":           1,
            "phase":              f"fixvar({fix_var_name})",
            "fix_var":            fix_var_name,
            "fix_var_range":      str(fix_var_range),
            "decomposition":      decomposition,
            "time_sec":           round(elapsed, 2),
            "trial":              trial_idx,
        }

    except StepTimeout as e:
        print(f"\n      !!! TIMEOUT (fixvar): {e}")
        raise

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"\n      !!! FIXVAR TRIAL FAILED: {type(e).__name__}: {e}")
        return {
            "formula_name":       eqn,
            "ground_truth":       true_formula,
            "rounded_expression": float('nan'),
            "rounded_r2":         float('nan'),
            "network_r2":         float('nan'),
            "r2_sym":             False,
            "symbolic_solution":  False,
            "noise_level":        NOISE_LEVEL,
            "invert_target":      "fixvar",
            "n_hidden":           1,
            "phase":              "fixvar",
            "fix_var":            fix_var_name if 'fix_var_name' in dir() else "N/A",
            "decomposition":      "N/A",
            "time_sec":           round(elapsed, 2),
            "trial":              trial_idx,
        }


# ============================================================================
# MAIN: run full pipeline for one formula
# ============================================================================

def run_formula_pipeline(eqn):
    """
    1. Load full data, auto-detect if fixing is needed.
    2. If no fix: try n_hidden=1 (up to max_trials), then n_hidden=2.
    3. If fix: auto-select var, auto-detect range, run fixvar pipeline (n_hidden=1).
    """
    row = meta[meta["Filename"] == eqn]
    if row.empty:
        print(f"  !!! '{eqn}' not found in metadata — skipping")
        return False, "N/A", None

    true_formula = row["Formula"].iloc[0]
    variable_headers = [c for c in row.columns
                        if c.startswith("v") and c.endswith("_name")]
    var_names_base = row[variable_headers].iloc[0].dropna().values.flatten().tolist()

    print(f"  True formula : {true_formula}")
    print(f"  Variables    : {var_names_base}")

    results_csv = RESULTS_DIR / f"{eqn}.csv"

    # ── Load full dataset for fix-variable detection ─────────────────────
    print(f"\n  Loading full dataset for variable analysis...")
    dataset_full = custom_data(
        eqn=eqn, logT=False, frac=1.0, shuffle=True, seed=SEED,
        positive_only=POSITIVE_ONLY, positive_on="inputs", data_dir=DATA_DIR,
    )
    loader_full = DataLoader(dataset_full, batch_size=len(dataset_full))
    x_full, y_full = next(iter(loader_full))
    X_FULL = x_full.numpy().astype(np.float32)
    y_FULL = y_full.squeeze().numpy().astype(np.float32)
    y_FULL_clean = y_FULL.copy()

    # ── Inject noise: ε ~ N(0, σ² · mean(y²)) ───────────────────────────
    if NOISE_LEVEL and float(NOISE_LEVEL) > 0:
        sigma = float(NOISE_LEVEL)
        y_rms = float(np.sqrt(np.mean(y_FULL.astype(np.float64) ** 2)))
        noise_std = sigma * y_rms
        rng_noise = np.random.default_rng(SEED)
        y_FULL = (y_FULL + rng_noise.normal(0.0, noise_std, size=y_FULL.shape)).astype(np.float32)
        print(f"  NOISE: σ={sigma}  RMS(y)={y_rms:.4g}  noise_std={noise_std:.4g}")

    global Y_FULL_CLEAN, X_FULL_CLEAN
    Y_FULL_CLEAN = y_FULL_clean.astype(np.float64)
    X_FULL_CLEAN = X_FULL

    n_total = X_FULL.shape[0]
    n_train = max(1, int(BASE_FRAC * n_total))
    print(f"  Full dataset: {n_total} samples, {len(var_names_base)} vars")
    print(f"  n_train = {n_train}  (base_frac={BASE_FRAC})")

    # ── Stage 0: should we fix a variable? ───────────────────────────────
    # (only if enable_fix_var is True)
    needs_fixing = False
    if ENABLE_FIX_VAR:
        print(f"\n  STAGE 0: Checking if variable fixing is needed...")
        fix_var_name, scan_scores, needs_fixing = select_variable_to_fix(
            X_FULL, y_FULL, var_names_base, scan_window_frac=SCAN_WINDOW_FRAC,
        )
    else:
        print(f"\n  STAGE 0: Fix-variable pipeline disabled (enable_fix_var=False)")
        print(f"           Using direct SMILE with n_hidden_sequence={N_HIDDEN_SEQUENCE}")
        fix_var_name = None

    # ──────────────────────────────────────────────────────────────────────
    # PATH A: Direct SMILE (no fix or fix-var disabled)
    # ──────────────────────────────────────────────────────────────────────
    if not needs_fixing:
        print(f"\n  → Running direct SMILE.")

        all_trial_results = []
        solved = False

        for depth_idx, n_hidden in enumerate(N_HIDDEN_SEQUENCE):
            print(f"\n  {'─'*60}")
            print(f"  DIRECT: n_hidden={n_hidden}  (depth {depth_idx+1}/{len(N_HIDDEN_SEQUENCE)})")
            print(f"  {'─'*60}")

            # When moving to a new depth, discard previous results so the
            # CSV reflects the current depth's (more exact) network/rounded R².
            if depth_idx > 0:
                all_trial_results = []

            for trial_idx in range(1, MAX_TRIALS + 1):
                print(f"\n  TRIAL {trial_idx}/{MAX_TRIALS}  (n_hidden={n_hidden})")
                trial_seed = SEED + trial_idx - 1

                try:
                    result = run_direct_trial(
                        eqn, true_formula, var_names_base,
                        X_FULL, y_FULL, n_train,
                        n_hidden, trial_idx, trial_seed,
                    )
                except StepTimeout:
                    # Skip remaining trials for this n_hidden, but still try HL=2.
                    print(f"\n  !!! Timeout on n_hidden={n_hidden} — "
                          f"skipping remaining trials at this depth")
                    break

                all_trial_results.append(result)

                if result["symbolic_solution"]:
                    print(f"\n  >>> SYMBOLIC SOLUTION on trial {trial_idx}! <<<")
                    for fill_idx in range(trial_idx + 1, MAX_TRIALS + 1):
                        filled = result.copy()
                        filled["trial"] = fill_idx
                        all_trial_results.append(filled)
                    solved = True
                    break

                r2_val = result["rounded_r2"]
                if (not np.isfinite(r2_val)) or r2_val < 0.9:
                    print(f"\n  R2={r2_val} < 0.9 — skipping remaining trials at "
                          f"n_hidden={n_hidden} (will try next depth if available)")
                    break

            # Persist progress after each depth
            if all_trial_results:
                pd.DataFrame(all_trial_results).to_csv(results_csv, index=False)

            if solved:
                return True, f"direct(n_hidden={n_hidden})", results_csv
            # If unsolved at this depth, continue to next depth in sequence.

        return False, "direct(unsolved)", results_csv

    # ──────────────────────────────────────────────────────────────────────
    # PATH B: Fix-variable pipeline (auto range detection)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n  → Variable fixing needed. Fix: {fix_var_name}")

    # Auto-compute fix point and fix range
    fix_var_idx = var_names_base.index(fix_var_name)
    fix_point = int(np.ceil(X_FULL[:, fix_var_idx].min()))
    print(f"    Fix point: {fix_point}  (ceil of min={X_FULL[:, fix_var_idx].min():.4f})")

    fix_var_range = find_fix_range_regression(
        X=X_FULL, y=y_FULL, var_names=var_names_base,
        fix_var_name=fix_var_name, fix_point=fix_point,
        max_pct=MAX_WINDOW_PCT, min_pct=MIN_WINDOW_PCT,
        other_range=OTHER_VAR_RANGE, constancy_thresh=CONSTANCY_THRESH,
    )
    print(f"    Auto-detected range: {fix_var_range}")

    trial_results = []
    solved = False
    formula_timed_out = False

    for trial_idx in range(1, MAX_TRIALS + 1):
        print(f"\n  TRIAL {trial_idx}/{MAX_TRIALS}  (fixvar: {fix_var_name})")
        trial_seed = SEED + trial_idx - 1

        try:
            result = run_fixvar_trial(
                eqn, true_formula, var_names_base,
                X_FULL, y_FULL, n_train,
                fix_var_name, fix_var_range, trial_idx, trial_seed,
            )
        except StepTimeout:
            print(f"\n  !!! Timeout — skipping formula")
            formula_timed_out = True
            break

        trial_results.append(result)

        if result["symbolic_solution"]:
            print(f"\n  >>> SYMBOLIC SOLUTION (fixvar) on trial {trial_idx}! <<<")
            for fill_idx in range(trial_idx + 1, MAX_TRIALS + 1):
                filled = result.copy()
                filled["trial"] = fill_idx
                trial_results.append(filled)
            solved = True
            break

        r2_val = result["rounded_r2"]
        if (not np.isfinite(r2_val)) or r2_val < 0.9:
            print(f"\n  R2={r2_val} < 0.9 — skipping remaining trials")
            break

    if trial_results:
        df = pd.DataFrame(trial_results)
        df.to_csv(results_csv, index=False)

    if formula_timed_out:
        return False, "timeout", results_csv

    if solved:
        return True, f"fixvar({fix_var_name})", results_csv

    return False, f"fixvar({fix_var_name},unsolved)", results_csv


# ============================================================================
# SINGLE-FORMULA MODE (called by subprocess)
# ============================================================================

if len(sys.argv) > 2 and sys.argv[2] == "--formula":
    _single_eqn = sys.argv[3]
    try:
        solved, best_cfg, csv_path = run_formula_pipeline(_single_eqn)
    except Exception as e:
        print(f"\n  !!! FORMULA {_single_eqn} CRASHED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0 if solved else 2)  # 0=solved, 2=not solved, 1=crash

# ============================================================================
# MAIN LOOP (parent process — spawns subprocesses)
# ============================================================================

import subprocess as _sp

summary_rows = []
total_start = time.time()

for formula_idx, eqn in enumerate(EQN_LIST):
    print(f"\n{'#'*70}")
    print(f"# FORMULA {formula_idx+1}/{len(EQN_LIST)}: {eqn}")
    print(f"{'#'*70}")

    # ── Resume logic: skip if results already exist ──────────────────
    existing_csv = RESULTS_DIR / f"{eqn}.csv"
    if existing_csv.exists():
        try:
            prev = pd.read_csv(existing_csv)
            if len(prev) > 0:
                prev_solved = prev["symbolic_solution"].any()
                print(f"  Already processed ({len(prev)} trials, solved={prev_solved}) — skipping")
                summary_rows.append({
                    "formula_name":     eqn,
                    "symbolic_solution": bool(prev_solved),
                    "noise_level":      NOISE_LEVEL,
                    "best_config":      "resumed",
                    "total_time_sec":   0,
                })
                continue
        except Exception:
            pass  # corrupted CSV — re-run

    formula_start = time.time()

    proc = _sp.run(
        [sys.executable, str(Path(__file__).resolve()), str(CONFIG_PATH), "--formula", eqn],
        timeout=3600,
    )

    formula_time = time.time() - formula_start

    if proc.returncode == 0:
        solved, best_cfg = True, "subprocess"
    elif proc.returncode == 2:
        solved, best_cfg = False, "unsolved"
    else:
        solved, best_cfg = False, f"crash(exit={proc.returncode})"
        print(f"\n  !!! FORMULA {eqn} CRASHED in subprocess (exit code {proc.returncode})")
        # Write NaN CSV so the formula is not retried on resume
        if not existing_csv.exists():
            row = meta[meta["Filename"] == eqn]
            if not row.empty:
                true_formula = row["Formula"].iloc[0]
            else:
                try:
                    true_formula = str(sp.sympify(eqn))
                except Exception:
                    true_formula = "N/A"
            crash_df = pd.DataFrame([{
                "formula_name":       eqn,
                "ground_truth":       true_formula,
                "rounded_expression": float('nan'),
                "rounded_r2":         float('nan'),
                "r2_sym":             False,
                    "symbolic_solution":  False,
                "noise_level":        NOISE_LEVEL,
                "invert_target":      "N/A",
                "n_hidden":           "N/A",
                "phase":              "crash",
                "time_sec":           round(formula_time, 2),
                "trial":              0,
            }])
            crash_df.to_csv(existing_csv, index=False)
            print(f"    Wrote crash CSV: {existing_csv}")

    # Read actual config from CSV if available
    if existing_csv.exists():
        try:
            prev = pd.read_csv(existing_csv)
            if len(prev) > 0 and prev["symbolic_solution"].any():
                solved = True
        except Exception:
            pass

    summary_rows.append({
        "formula_name":     eqn,
        "symbolic_solution": solved,
        "noise_level":      NOISE_LEVEL,
        "best_config":      best_cfg,
        "total_time_sec":   round(formula_time, 2),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    status = "SOLVED" if solved else "NOT SOLVED"
    print(f"\n  [{status}]  {eqn}  ({formula_time:.1f}s)  config={best_cfg}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = time.time() - total_start

print(f"\n{'#'*70}")
print("# SMILE PIPELINE COMPLETE")
print(f"{'#'*70}")

summary_df = pd.DataFrame(summary_rows)
n_solved = summary_df["symbolic_solution"].sum()
n_total = len(summary_df)

print(f"\n  Solved: {n_solved}/{n_total}")
print(f"  Total time: {total_time:.1f}s")
print(f"\n  Summary CSV: {summary_csv}")
print(f"\n{summary_df.to_string(index=False)}")

print(f"\n{'#'*70}")
print("DONE!")
print(f"{'#'*70}")
