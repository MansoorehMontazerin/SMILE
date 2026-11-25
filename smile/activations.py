"""
- log:  lower cutoff at LOG_LOWER_CUTOFF (1e-5)
- exp:  upper cutoff at EXP_UPPER_CUTOFF (50.0)
- mult: input lower cutoff at MULT_INPUT_CUTOFF (1e-5),
                 inner-product upper cutoff at MULT_INNER_CUTOFF (50.0)
WARNING: The cutoff constants (LOG_LOWER_CUTOFF, EXP_UPPER_CUTOFF,
MULT_INPUT_CUTOFF, MULT_INNER_CUTOFF) are defined as module-level globals.
They are NOT guaranteed to remain unchanged at runtime. Any other module
that imports them can reassign them (e.g.,
``activations.EXP_UPPER_CUTOFF = 10.0``), and since the functions read
the globals at call time, such reassignment WILL silently change the
behavior of every activation function in this module. Additionally,
nothing prevents a monkey-patch or a config loader from overwriting
these values. If immutability is required, consider:
- Making them ``Final`` annotations (for static checkers only, not enforced
    at runtime).
- Wrapping them in a frozen dataclass or ``types.MappingProxyType``.
- Passing them explicitly as function parameters with the current globals
    as default values (captured at definition time, not call time).
As written, the constants are authoritative only by convention; no
mechanism in this file prevents them from being overwritten elsewhere.
Custom activation functions for the SMILE network.

SMILE = Sin, Multiplication, Identity, Log, Exp

Each activation has cutoffs for numerical stability:
  - log:  lower cutoff at LOG_LOWER_CUTOFF (1e-4)
  - exp:  upper cutoff at EXP_UPPER_CUTOFF (4.0)
  - mult: input lower cutoff at MULT_INPUT_CUTOFF (1e-4),
           inner-product upper cutoff at MULT_INNER_CUTOFF (4.0)

Beyond a cutoff the output is held constant (gradient is zero there),
and a penalty proportional to the violation magnitude is returned so
the loss encourages the network to stay in the legal ranges.
"""

import torch

# ── Cutoff constants ──────────────────────────────────────────────────────────
LOG_LOWER_CUTOFF   = 1e-4   # log(x) for x < cutoff → clamped
EXP_UPPER_CUTOFF   = 5.0    # exp(x) for x > cutoff → clamped
MULT_INPUT_CUTOFF  = 1e-4   # x_i fed to mult neuron must be > cutoff
MULT_INNER_CUTOFF  = 5.0    # inner product before exp in mult clamped


# ── Individual activation helpers ─────────────────────────────────────────────

def safe_log_activation(x: torch.Tensor):
    """
    log(x) with lower cutoff.

    Returns
    -------
    out     : tensor, same shape as x
    penalty : scalar tensor (sum of violations)
    """
    violation = torch.clamp(LOG_LOWER_CUTOFF - x, min=0.0)
    penalty   = violation.sum()
    out       = torch.log(torch.clamp(x, min=LOG_LOWER_CUTOFF))
    return out, penalty


def safe_exp_activation(x: torch.Tensor):
    """
    exp(x) with upper cutoff.

    Returns
    -------
    out     : tensor, same shape as x
    penalty : scalar tensor
    """
    violation = torch.clamp(x - EXP_UPPER_CUTOFF, min=0.0)
    penalty   = violation.sum()
    out       = torch.exp(torch.clamp(x, max=EXP_UPPER_CUTOFF))
    return out, penalty


def mult_activation(x: torch.Tensor, weight: torch.Tensor):
    """
    Multiplication (monomial) activation.

    Computes  exp( weight @ log(x) )  ==  prod_i  x_i^{weight_i}
    Applied to a batch of inputs.

    Parameters
    ----------
    x      : (batch, d_in)
    weight : (n_neurons, d_in)  – rows are per-neuron weight vectors

    Returns
    -------
    out     : (batch, n_neurons)
    penalty : scalar tensor
    """
    # Penalty for x going below the positive cutoff
    violation_in  = torch.clamp(MULT_INPUT_CUTOFF - x, min=0.0)
    penalty_in    = violation_in.sum()
    x_safe        = torch.clamp(x, min=MULT_INPUT_CUTOFF)

    log_x  = torch.log(x_safe)                  # (batch, d_in)
    inner  = log_x @ weight.t()                  # (batch, n_neurons)

    # Penalty for inner product exceeding the exp cutoff
    violation_out = torch.clamp(inner - MULT_INNER_CUTOFF, min=0.0)
    penalty_out   = violation_out.sum()
    out           = torch.exp(torch.clamp(inner, max=MULT_INNER_CUTOFF))

    return out, penalty_in + penalty_out
