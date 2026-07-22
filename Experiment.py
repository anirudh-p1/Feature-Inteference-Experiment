"""
Attention Kernels and Feature Interference Under a Representational Bottleneck
================================================================================
Run this top to bottom in Google Colab (Runtime > Run all). No GPU needed —
everything here trains in seconds. Requires: torch, numpy, scikit-learn
(all pre-installed on Colab).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")  # sklearn warns on tiny/imbalanced folds; we handle that ourselves

# ============================================================================
# 1. DATA GENERATION
# ============================================================================
# Everything below is invented, synthetic data. There is no file to download —
# we are building a small fake universe of 20 yes/no "concepts" where WE
# control exactly how they relate to each other. That's what lets us measure
# things precisely later (we always know the ground truth).

N_FEATURES = 20

def generate_dataset_A(n, seed, p=0.1):
    """Pure Independent dataset: all 20 features on/off completely separately."""
    rng = np.random.default_rng(seed)
    X = (rng.random((n, N_FEATURES)) < p).astype(np.float32)
    return X

def generate_dataset_B(n, seed):
    """
    Structured dataset:
      features 0-7   : independent, p=0.1 each (control group)
      features 8-13  : three POSITIVE-correlated pairs (8,9) (10,11) (12,13)
      features 14-19 : three MUTUALLY EXCLUSIVE pairs (14,15) (16,17) (18,19)
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    # independent control features
    X[:, 0:8] = (rng.random((n, 8)) < 0.1).astype(np.float32)

    # positive-correlated pairs: A ~ Bernoulli(0.15), B | A=1 ~ 0.8, B | A=0 ~ 0.05
    for a_idx, b_idx in [(8, 9), (10, 11), (12, 13)]:
        a = (rng.random(n) < 0.15).astype(np.float32)
        b_prob = np.where(a == 1, 0.8, 0.05)
        b = (rng.random(n) < b_prob).astype(np.float32)
        X[:, a_idx] = a
        X[:, b_idx] = b

    # mutually exclusive pairs: A ~ Bernoulli(0.15), B | A=1 = 0, B | A=0 ~ 0.15
    for a_idx, b_idx in [(14, 15), (16, 17), (18, 19)]:
        a = (rng.random(n) < 0.15).astype(np.float32)
        b_prob = np.where(a == 1, 0.0, 0.15)
        b = (rng.random(n) < b_prob).astype(np.float32)
        X[:, a_idx] = a
        X[:, b_idx] = b

    return X

# Feature group definitions used later for group-wise metrics on Dataset B
GROUP_INDEPENDENT = list(range(0, 8))
GROUP_POSITIVE_PAIRS = [(8, 9), (10, 11), (12, 13)]
GROUP_EXCLUSIVE_PAIRS = [(14, 15), (16, 17), (18, 19)]


# ============================================================================
# 2. THE THREE ATTENTION KERNELS
# ============================================================================
# All three take the SAME raw similarity scores. Only the function that turns
# a score into a normalized weight changes. Everything else in the model
# (embeddings, pooling, reconstruction head, training) is identical across
# all three, so any measured difference is attributable to this one swap.

def kernel_softmax(scores):
    """
    APPROACH 1 — Softmax (baseline).
    What it does: standard exponential normalization, exactly what real
    Transformers use.
    How: weight_ij = exp(score_ij) / sum_k exp(score_ik).
    What we expect: the ratio between the strongest and weakest attended
    position is UNBOUNDED — a small score difference can become an enormous
    weight difference. This gives the model maximum flexibility to develop
    very sharp, winner-take-all attention, which is one route by which
    tangled, hard-to-audit representations can form during training. This is
    our point of comparison, not something we're trying to prove is "bad."
    """
    return torch.softmax(scores, dim=-1)


def kernel_bounded(scores, floor=0.05):
    """
    APPROACH 2 — Bounded sigmoid + floor (our main intervention).
    What it does: replaces the unbounded exponential with a saturating
    sigmoid, then adds a small constant floor before normalizing.
    How: weight_ij = (sigmoid(score_ij) + floor) / sum_k (sigmoid(score_ik) + floor).
    What we expect: the floor guarantees every raw weight stays above a fixed
    positive value, capping the max/min weight ratio at roughly (1+floor)/floor
    (~21 here) no matter how extreme the underlying scores get. If attention
    concentration is part of what allows harmful feature interference to
    form, capping it should show up as less interference on features that
    have nothing to do with each other. Note: unlike softmax, this kernel is
    NOT translation-invariant (shifting all scores by a constant does change
    the output), and it pulls toward uniform attention as scores go very
    negative — this is a real difference from softmax beyond "boundedness"
    alone, and we say so plainly rather than pretending the isolation is perfect.
    """
    s = torch.sigmoid(scores) + floor
    return s / s.sum(dim=-1, keepdim=True)


def kernel_sparse(scores):
    """
    APPROACH 3 — Squared-ReLU hard-sparse (contrast condition).
    What it does: any position with a negative similarity score gets EXACTLY
    zero weight — a hard cutoff, not an asymptotic approach to zero like the
    other two kernels.
    How: weight_ij = relu(score_ij)^2 / sum_k relu(score_ik)^2. If every
    score in a row is negative, the denominator would be zero (divide-by-zero
    risk) — we fall back to pure self-attention (weight on itself = 1,
    everything else = 0) in that case, which is a safe, well-defined choice.
    What we expect: this tests a different, harder kind of sparsity than
    kernel 2's smooth cap. If hard cutoffs help more than smooth bounding, it
    tells us the ACTIVE INGREDIENT is eliminating weak interactions entirely
    (not merely limiting their magnitude). It could also be too aggressive
    and lose useful weak signal — either result is informative.
    This is NOT the sparsemax operator — it's a simpler hard-sparse
    approximation, and we describe it as such, not as equivalent to sparsemax.
    """
    r = torch.relu(scores) ** 2
    denom = r.sum(dim=-1, keepdim=True)
    n = scores.shape[-1]
    eye = torch.eye(n, device=scores.device).unsqueeze(0)  # (1, n, n)
    zero_rows = (denom == 0).float()                        # (batch, n, 1)
    safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    weights = r / safe_denom
    weights = weights * (1 - zero_rows) + eye * zero_rows
    return weights


KERNELS = {
    "softmax": kernel_softmax,
    "bounded": kernel_bounded,
    "sparse": kernel_sparse,
}


# ============================================================================
# 3. MODEL
# ============================================================================
class ToyAttentionModel(nn.Module):
    """
    x (20-dim binary)
       -> per-slot embedding: X_i = x_i * e_i        (20 slots, each own 8-dim embedding)
       -> self-attention across the 20 slots         (kernel swapped here)
       -> sum the 20 updated vectors -> z (bottleneck, hidden_dim-dim)
       -> linear + sigmoid -> x_hat (20 predicted probabilities)
    """
    def __init__(self, n_features=N_FEATURES, hidden_dim=8, kernel="softmax"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_fn = KERNELS[kernel]
        self.slot_embed = nn.Parameter(torch.randn(n_features, hidden_dim) * 0.1)
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.readout = nn.Linear(hidden_dim, n_features)

    def forward(self, x, return_attn=False):
        # x: (batch, n_features)
        tok = x.unsqueeze(-1) * self.slot_embed.unsqueeze(0)   # (batch, n_features, hidden_dim)
        q, k, v = self.Wq(tok), self.Wk(tok), self.Wv(tok)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_dim ** 0.5)  # (batch, n, n)
        weights = self.kernel_fn(scores)
        attn_out = torch.matmul(weights, v)                    # (batch, n_features, hidden_dim)
        z = attn_out.sum(dim=1)                                 # (batch, hidden_dim) -- the bottleneck
        x_hat = torch.sigmoid(self.readout(z))
        if return_attn:
            return x_hat, z, weights
        return x_hat, z


# ============================================================================
# 4. TRAINING
# ============================================================================
def train_model(X_train, hidden_dim, kernel, seed, steps=2000, batch_size=256, lr=1e-3, log_every=100):
    torch.manual_seed(seed)
    model = ToyAttentionModel(hidden_dim=hidden_dim, kernel=kernel)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    X_train_t = torch.tensor(X_train)
    n = X_train_t.shape[0]
    loss_curve = []

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        batch = X_train_t[idx]
        x_hat, z = model(batch)
        loss = loss_fn(x_hat, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            loss_curve.append((step, loss.item()))
    return model, loss_curve


def eval_reconstruction_loss(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X)
        x_hat, z = model(X_t)
        loss = nn.BCELoss()(x_hat, X_t).item()
    model.train()
    return loss


# ============================================================================
# 5. METRICS
# ============================================================================
def get_bottleneck(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X)
        _, z = model(X_t)
    model.train()
    return z.numpy()


def ridge_interference(model, X_probe_train, groups):
    """
    PRIMARY METRIC A — regression-based feature direction.
    Fit z ~ X via ridge regression (ridge, not plain least squares, because
    correlated features make the design matrix poorly conditioned). Each
    column of the fitted coefficient matrix, transposed, gives an ESTIMATED
    LINEAR feature direction per ground-truth feature (an approximation --
    the true mapping through attention + nonlinearities is not actually
    linear). We then measure how much these estimated directions overlap
    (cosine similarity) within each relationship group.
    """
    Z = get_bottleneck(model, X_probe_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_probe_train, Z)
    D = ridge.coef_.T  # (n_features, hidden_dim)
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    D_norm = D / norms
    cos_sim = D_norm @ D_norm.T  # (n_features, n_features)

    results = {}
    results["independent_mean_abs_cos"] = _group_mean_abs(cos_sim, groups["independent"])
    results["positive_pairs_mean_abs_cos"] = _pair_mean_abs(cos_sim, groups["positive_pairs"])
    results["exclusive_pairs_mean_abs_cos"] = _pair_mean_abs(cos_sim, groups["exclusive_pairs"])
    return results, cos_sim


def _group_mean_abs(cos_sim, indices):
    vals = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            vals.append(abs(cos_sim[indices[i], indices[j]]))
    return float(np.mean(vals)) if vals else float("nan")


def _pair_mean_abs(cos_sim, pairs):
    vals = [abs(cos_sim[a, b]) for a, b in pairs]
    return float(np.mean(vals)) if vals else float("nan")


def intervention_direction(model, X_baseline, groups, n_baselines=200):
    """
    PRIMARY METRIC B — intervention-based feature direction.
    For each ground-truth feature i: take a batch of real baseline examples,
    force feature i OFF in one copy and ON in another (holding every other
    feature exactly as in the baseline example), pass both through the
    FROZEN model, and record the shift in the bottleneck representation.
    Averaged over many baselines, this gives an empirical direction v_i that
    reflects what ACTUALLY happens when you change feature i -- unlike the
    ridge metric above, this can't be fooled by two features merely
    co-occurring in the data without the model actually treating them
    similarly.
    """
    rng = np.random.default_rng(0)
    base_idx = rng.choice(len(X_baseline), size=min(n_baselines, len(X_baseline)), replace=False)
    base = X_baseline[base_idx].copy()

    directions = np.zeros((N_FEATURES, model.hidden_dim))
    for i in range(N_FEATURES):
        on = base.copy();  on[:, i] = 1.0
        off = base.copy(); off[:, i] = 0.0
        z_on = get_bottleneck(model, on)
        z_off = get_bottleneck(model, off)
        directions[i] = (z_on - z_off).mean(axis=0)

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    D_norm = directions / norms
    cos_sim = D_norm @ D_norm.T

    results = {}
    results["independent_mean_abs_cos"] = _group_mean_abs(cos_sim, groups["independent"])
    results["positive_pairs_mean_abs_cos"] = _pair_mean_abs(cos_sim, groups["positive_pairs"])
    results["exclusive_pairs_mean_abs_cos"] = _pair_mean_abs(cos_sim, groups["exclusive_pairs"])
    return results, cos_sim


def probe_recovery(model, X_probe_train, X_probe_eval):
    """
    SECONDARY METRIC — feature recoverability via linear probes.
    One logistic-regression probe per ground-truth feature, trained on
    bottleneck vectors from the PROBE-TRAIN pool and evaluated on the
    PROBE-EVAL pool (completely separate from the network's own training
    and test data -- reusing data here would be leakage and would inflate
    these numbers). We report AUROC and balanced accuracy rather than raw
    accuracy, since with ~10% activation frequency a trivial "always predict
    off" probe would otherwise score ~90% while being useless.
    """
    Z_train = get_bottleneck(model, X_probe_train)
    Z_eval = get_bottleneck(model, X_probe_eval)

    aurocs, bal_accs = [], []
    for i in range(N_FEATURES):
        y_train = X_probe_train[:, i]
        y_eval = X_probe_eval[:, i]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
            continue  # can't score a probe meaningfully with only one class present
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z_train, y_train)
        probs = clf.predict_proba(Z_eval)[:, 1]
        preds = clf.predict(Z_eval)
        aurocs.append(roc_auc_score(y_eval, probs))
        bal_accs.append(balanced_accuracy_score(y_eval, preds))
    return {
        "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "mean_balanced_acc": float(np.mean(bal_accs)) if bal_accs else float("nan"),
        "n_features_scored": len(aurocs),
    }


def mechanistic_stats(model, X):
    """
    MECHANISTIC CHECK -- do the kernels actually behave the way their
    formulas suggest? We measure this directly rather than assuming it,
    because e.g. squared-ReLU might turn out not to be meaningfully sparse
    in practice if raw scores are mostly positive.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X)
        _, z, weights = model(X_t, return_attn=True)  # weights: (batch, n, n)
        w = weights.numpy()
    model.train()

    eps = 1e-12
    entropy = -(w * np.log(w + eps)).sum(axis=-1).mean()
    max_weight = w.max(axis=-1).mean()
    effective_support = (1.0 / (w ** 2).sum(axis=-1)).mean()
    frac_exact_zero = (w == 0).mean()

    return {
        "attention_entropy": float(entropy),
        "max_attention_weight": float(max_weight),
        "effective_support_size": float(effective_support),
        "fraction_exact_zero_weights": float(frac_exact_zero),
    }


# ============================================================================
# 6. CALIBRATION STEP -- run this first
# ============================================================================
def run_calibration(steps=1000):
    print("=" * 70)
    print("CALIBRATION: softmax kernel, Dataset A, varying hidden_dim")
    print("(we want: reasonable reconstruction loss AND visible interference)")
    print("=" * 70)
    X_train = generate_dataset_A(10000, seed=1)
    X_test = generate_dataset_A(2000, seed=2)
    groups = {"independent": list(range(N_FEATURES)), "positive_pairs": [], "exclusive_pairs": []}

    for dim in [4, 8, 16, 20]:
        model, _ = train_model(X_train, hidden_dim=dim, kernel="softmax", seed=0, steps=steps)
        loss = eval_reconstruction_loss(model, X_test)
        interference, _ = ridge_interference(model, X_train, groups)
        print(f"  hidden_dim={dim:2d}  |  test BCE={loss:.4f}  |  mean|cos| among independent features={interference['independent_mean_abs_cos']:.4f}")
    print()


# ============================================================================
# 7. MAIN EXPERIMENT -- 3 kernels x 2 datasets x 3 seeds
# ============================================================================
def run_main_experiment(hidden_dim=8, steps=2000, seeds=(0, 1, 2)):
    groups_B = {
        "independent": GROUP_INDEPENDENT,
        "positive_pairs": GROUP_POSITIVE_PAIRS,
        "exclusive_pairs": GROUP_EXCLUSIVE_PAIRS,
    }
    groups_A = {"independent": list(range(N_FEATURES)), "positive_pairs": [], "exclusive_pairs": []}

    all_results = []

    for dataset_name, gen_fn, groups in [("A_independent", generate_dataset_A, groups_A),
                                          ("B_structured", generate_dataset_B, groups_B)]:
        for kernel_name in KERNELS:
            for seed in seeds:
                X_train = gen_fn(10000, seed=seed * 10 + 1)
                X_test = gen_fn(2000, seed=seed * 10 + 2)
                X_probe_train = gen_fn(5000, seed=seed * 10 + 3)
                X_probe_eval = gen_fn(2000, seed=seed * 10 + 4)

                model, loss_curve = train_model(X_train, hidden_dim, kernel_name, seed, steps=steps)

                recon_loss = eval_reconstruction_loss(model, X_test)
                ridge_res, _ = ridge_interference(model, X_probe_train, groups)
                interv_res, _ = intervention_direction(model, X_probe_train, groups)
                probe_res = probe_recovery(model, X_probe_train, X_probe_eval)
                mech_res = mechanistic_stats(model, X_test)

                row = {
                    "dataset": dataset_name,
                    "kernel": kernel_name,
                    "seed": seed,
                    "recon_loss": recon_loss,
                    **{f"ridge_{k}": v for k, v in ridge_res.items()},
                    **{f"interv_{k}": v for k, v in interv_res.items()},
                    **probe_res,
                    **mech_res,
                }
                all_results.append(row)
                print(f"[{dataset_name:14s} | {kernel_name:8s} | seed={seed}] "
                      f"recon={recon_loss:.4f}  AUROC={probe_res['mean_auroc']:.3f}  "
                      f"indep|cos|={ridge_res['independent_mean_abs_cos']:.3f}")

    return all_results


def summarize(all_results):
    import pandas as pd
    df = pd.DataFrame(all_results)
    numeric_cols = [c for c in df.columns if c not in ("dataset", "kernel", "seed")]
    summary = df.groupby(["dataset", "kernel"])[numeric_cols].agg(["mean", "std"])
    return df, summary


if __name__ == "__main__":
    run_calibration(steps=1000)
    results = run_main_experiment(hidden_dim=8, steps=2000, seeds=(0, 1, 2))
    df, summary = summarize(results)
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across 3 seeds)")
    print("=" * 70)
    print(summary)
    df.to_csv("results_raw.csv", index=False)
    print("\nRaw per-run results saved to results_raw.csv")
