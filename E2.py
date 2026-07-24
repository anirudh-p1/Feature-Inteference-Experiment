# ============================================================================
# SERIOUS 2×2 EXPERIMENT:
# ATTENTION KERNEL × BOTTLENECK CAPACITY
# ============================================================================
#
# MAIN FACTORIAL DESIGN
# ---------------------
#
# Factor 1: Attention kernel
#   - softmax
#   - bounded sigmoid + floor
#
# Factor 2: Bottleneck capacity
#   - tight: hidden_dim = 8
#   - wide:  hidden_dim = 20
#
# Formal 2×2 cells:
#
#   1. softmax / tight
#   2. softmax / wide
#   3. bounded / tight
#   4. bounded / wide
#
# Additional contrast condition:
#
#   5. sparse / tight
#   6. sparse / wide
#
# The sparse condition is NOT part of the formal 2×2 factorial analysis.
# It is included as a separate contrast because hard sparsity is qualitatively
# different from both softmax and bounded attention.
#
# MAIN SCIENTIFIC QUESTION
# ------------------------
#
# Does the effect of the attention kernel depend on bottleneck pressure?
#
# If bounded attention only helps at hidden_dim=8, then the effect may be
# specifically related to compression pressure.
#
# If bounded attention helps at both hidden dimensions, that suggests a more
# general representational effect.
#
# If the effect disappears at hidden_dim=20, that suggests the bottleneck,
# rather than the attention kernel, is the primary driver of interference.
#
# ============================================================================
#
# REQUIRED PACKAGES
# -----------------
# torch
# numpy
# pandas
# scikit-learn
#
# No GPU required.
# ============================================================================


import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)

warnings.filterwarnings("ignore")


# ============================================================================
# 1. GLOBAL EXPERIMENT CONFIGURATION
# ============================================================================

N_FEATURES = 20

# Formal 2×2 factorial design.
BOTTLENECKS = {
    "tight": 8,
    "wide": 20,
}

FACTORIAL_KERNELS = [
    "softmax",
    "bounded",
]

# Separate contrast condition.
CONTRAST_KERNELS = [
    "sparse",
]

# All kernels used in the full experiment.
ALL_KERNELS = (
    FACTORIAL_KERNELS
    + CONTRAST_KERNELS
)

# Multiple independent random seeds.
SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

# Training configuration.
TRAIN_STEPS = 2500
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# Intervention configuration.
N_INTERVENTION_BASELINES = 500

# Fixed data sizes.
N_TRAIN = 10000
N_TEST = 3000
N_PROBE_TRAIN = 5000
N_PROBE_EVAL = 3000


# ============================================================================
# 2. REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    """
    Set all relevant random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# 3. DATA GENERATION
# ============================================================================

GROUP_INDEPENDENT = list(range(0, 8))

GROUP_POSITIVE_PAIRS = [
    (8, 9),
    (10, 11),
    (12, 13),
]

GROUP_EXCLUSIVE_PAIRS = [
    (14, 15),
    (16, 17),
    (18, 19),
]


def generate_dataset_A(
    n,
    seed,
    p=0.10,
):
    """
    Dataset A:
    All 20 binary features are mutually independent.
    """
    rng = np.random.default_rng(seed)

    X = (
        rng.random(
            (
                n,
                N_FEATURES,
            )
        )
        < p
    ).astype(np.float32)

    return X


def generate_dataset_B(
    n,
    seed,
):
    """
    Dataset B:
      features 0-7:
          independent controls

      features 8-13:
          three positively associated pairs

      features 14-19:
          three mutually exclusive pairs
    """

    rng = np.random.default_rng(seed)

    X = np.zeros(
        (
            n,
            N_FEATURES,
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Independent control features.
    # --------------------------------------------------------

    X[:, 0:8] = (
        rng.random(
            (
                n,
                8,
            )
        )
        < 0.10
    ).astype(np.float32)

    # --------------------------------------------------------
    # Positively associated pairs.
    # --------------------------------------------------------

    for a_idx, b_idx in GROUP_POSITIVE_PAIRS:

        a = (
            rng.random(n)
            < 0.15
        ).astype(np.float32)

        b_probability = np.where(
            a == 1,
            0.80,
            0.05,
        )

        b = (
            rng.random(n)
            < b_probability
        ).astype(np.float32)

        X[:, a_idx] = a
        X[:, b_idx] = b

    # --------------------------------------------------------
    # Mutually exclusive pairs.
    # --------------------------------------------------------

    for a_idx, b_idx in GROUP_EXCLUSIVE_PAIRS:

        a = (
            rng.random(n)
            < 0.15
        ).astype(np.float32)

        b_probability = np.where(
            a == 1,
            0.0,
            0.15,
        )

        b = (
            rng.random(n)
            < b_probability
        ).astype(np.float32)

        X[:, a_idx] = a
        X[:, b_idx] = b

    return X


# ============================================================================
# 4. EMPIRICAL DATASET VALIDATION
# ============================================================================

def empirical_pair_statistics(
    X,
    pairs,
):
    """
    Verify the actual relationships produced by the generator.
    """

    rows = []

    for i, j in pairs:

        xi = X[:, i]
        xj = X[:, j]

        rows.append(
            {
                "feature_i": i,
                "feature_j": j,
                "p_i": xi.mean(),
                "p_j": xj.mean(),
                "pearson_corr": np.corrcoef(
                    xi,
                    xj,
                )[0, 1],
                "coactivation_rate": np.mean(
                    (
                        xi == 1
                    )
                    &
                    (
                        xj == 1
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def print_dataset_validation():

    print("=" * 80)
    print("EMPIRICAL DATASET VALIDATION")
    print("=" * 80)

    X = generate_dataset_B(
        n=100000,
        seed=12345,
    )

    print("\nPositive pairs:")
    print(
        empirical_pair_statistics(
            X,
            GROUP_POSITIVE_PAIRS,
        ).to_string(
            index=False
        )
    )

    print("\nExclusive pairs:")
    print(
        empirical_pair_statistics(
            X,
            GROUP_EXCLUSIVE_PAIRS,
        ).to_string(
            index=False
        )
    )

    print()


# ============================================================================
# 5. ATTENTION KERNELS
# ============================================================================

def kernel_softmax(
    scores,
    temperature=1.0,
):
    """
    Standard temperature-controlled softmax.
    """

    return torch.softmax(
        scores / temperature,
        dim=-1,
    )


def kernel_bounded(
    scores,
    floor=0.05,
):
    """
    Smooth bounded attention.

    The sigmoid compresses score differences.
    The floor prevents exact zero weights.

    Important:
    Unlike softmax, this is not translation-invariant.
    It is therefore a distinct normalization family, not merely a capped
    softmax.
    """

    raw = (
        torch.sigmoid(scores)
        + floor
    )

    return (
        raw
        / raw.sum(
            dim=-1,
            keepdim=True,
        )
    )


def kernel_sparse(
    scores,
):
    """
    Hard-sparse squared-ReLU attention.

    Negative scores receive exactly zero weight.

    If an entire row is inactive, fall back to self-attention.
    """

    raw = (
        torch.relu(scores)
        ** 2
    )

    denominator = raw.sum(
        dim=-1,
        keepdim=True,
    )

    n = scores.shape[-1]

    identity = torch.eye(
        n,
        device=scores.device,
        dtype=scores.dtype,
    ).unsqueeze(0)

    zero_rows = (
        denominator
        == 0
    )

    safe_denominator = torch.where(
        zero_rows,
        torch.ones_like(
            denominator
        ),
        denominator,
    )

    weights = (
        raw
        / safe_denominator
    )

    weights = torch.where(
        zero_rows,
        identity,
        weights,
    )

    return weights


# ============================================================================
# 6. MODEL
# ============================================================================

class ToyAttentionModel(
    nn.Module
):

    def __init__(
        self,
        hidden_dim,
        kernel,
        n_features=N_FEATURES,
        softmax_temperature=1.0,
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.kernel_name = kernel
        self.n_features = n_features
        self.softmax_temperature = (
            softmax_temperature
        )

        # One learned embedding per input feature.
        self.slot_embed = nn.Parameter(
            torch.randn(
                n_features,
                hidden_dim,
            )
            * 0.1
        )

        self.Wq = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.Wk = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.Wv = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.readout = nn.Linear(
            hidden_dim,
            n_features,
        )

    def apply_kernel(
        self,
        scores,
    ):

        if (
            self.kernel_name
            == "softmax"
        ):

            return kernel_softmax(
                scores,
                temperature=(
                    self.softmax_temperature
                ),
            )

        if (
            self.kernel_name
            == "bounded"
        ):

            return kernel_bounded(
                scores
            )

        if (
            self.kernel_name
            == "sparse"
        ):

            return kernel_sparse(
                scores
            )

        raise ValueError(
            f"Unknown kernel: "
            f"{self.kernel_name}"
        )

    def forward(
        self,
        x,
        return_attn=False,
    ):

        # ----------------------------------------------------
        # x:
        #   (batch, n_features)
        #
        # tokens:
        #   (batch, n_features, hidden_dim)
        # ----------------------------------------------------

        tokens = (
            x.unsqueeze(-1)
            * self.slot_embed.unsqueeze(0)
        )

        q = self.Wq(
            tokens
        )

        k = self.Wk(
            tokens
        )

        v = self.Wv(
            tokens
        )

        scores = torch.matmul(
            q,
            k.transpose(
                -1,
                -2,
            ),
        ) / np.sqrt(
            self.hidden_dim
        )

        weights = self.apply_kernel(
            scores
        )

        attended = torch.matmul(
            weights,
            v
        )

        # ----------------------------------------------------
        # Sum pooling creates the bottleneck.
        # ----------------------------------------------------

        z = attended.sum(
            dim=1
        )

        x_hat = torch.sigmoid(
            self.readout(
                z
            )
        )

        if return_attn:

            return (
                x_hat,
                z,
                weights,
            )

        return (
            x_hat,
            z,
        )


# ============================================================================
# 7. TRAINING
# ============================================================================

def train_model(
    X_train,
    hidden_dim,
    kernel,
    seed,
    steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    softmax_temperature=1.0,
):

    set_seed(
        seed
    )

    model = ToyAttentionModel(
        hidden_dim=hidden_dim,
        kernel=kernel,
        softmax_temperature=(
            softmax_temperature
        ),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_fn = nn.BCELoss()

    X_train_t = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    n = len(
        X_train_t
    )

    model.train()

    for step in range(
        steps
    ):

        indices = torch.randint(
            0,
            n,
            (
                batch_size,
            ),
        )

        batch = (
            X_train_t[
                indices
            ]
        )

        x_hat, _ = model(
            batch
        )

        loss = loss_fn(
            x_hat,
            batch,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return model


# ============================================================================
# 8. BOTTLENECK EXTRACTION
# ============================================================================

def get_bottleneck(
    model,
    X,
):

    model.eval()

    with torch.no_grad():

        X_t = torch.tensor(
            X,
            dtype=torch.float32,
        )

        _, z = model(
            X_t
        )

    model.train()

    return z.numpy()


# ============================================================================
# 9. RECONSTRUCTION METRICS
# ============================================================================

def evaluate_reconstruction(
    model,
    X,
):

    model.eval()

    with torch.no_grad():

        X_t = torch.tensor(
            X,
            dtype=torch.float32,
        )

        x_hat, _ = model(
            X_t
        )

        bce = nn.BCELoss()(
            x_hat,
            X_t,
        ).item()

        predictions = (
            x_hat.cpu().numpy()
            >= 0.5
        ).astype(
            np.float32
        )

    model.train()

    balanced_accuracies = []

    for i in range(
        N_FEATURES
    ):

        balanced_accuracies.append(
            balanced_accuracy_score(
                X[:, i],
                predictions[:, i],
            )
        )

    return {

        "recon_bce":
            float(
                bce
            ),

        "mean_balanced_accuracy":
            float(
                np.mean(
                    balanced_accuracies
                )
            ),
    }


# ============================================================================
# 10. LINEAR PROBE RECOVERY
# ============================================================================

def probe_recovery(
    model,
    X_probe_train,
    X_probe_eval,
):

    Z_train = get_bottleneck(
        model,
        X_probe_train,
    )

    Z_eval = get_bottleneck(
        model,
        X_probe_eval,
    )

    aurocs = []
    average_precisions = []
    balanced_accuracies = []

    for i in range(
        N_FEATURES
    ):

        y_train = (
            X_probe_train[
                :,
                i
            ]
        )

        y_eval = (
            X_probe_eval[
                :,
                i
            ]
        )

        if (
            len(
                np.unique(
                    y_train
                )
            )
            < 2
        ):

            continue

        if (
            len(
                np.unique(
                    y_eval
                )
            )
            < 2
        ):

            continue

        clf = LogisticRegression(
            max_iter=1000
        )

        clf.fit(
            Z_train,
            y_train,
        )

        probabilities = (
            clf.predict_proba(
                Z_eval
            )[
                :,
                1
            ]
        )

        predictions = clf.predict(
            Z_eval
        )

        aurocs.append(
            roc_auc_score(
                y_eval,
                probabilities,
            )
        )

        average_precisions.append(
            average_precision_score(
                y_eval,
                probabilities,
            )
        )

        balanced_accuracies.append(
            balanced_accuracy_score(
                y_eval,
                predictions,
            )
        )

    return {

        "mean_auroc":
            float(
                np.mean(
                    aurocs
                )
            ),

        "mean_average_precision":
            float(
                np.mean(
                    average_precisions
                )
            ),

        "mean_probe_balanced_accuracy":
            float(
                np.mean(
                    balanced_accuracies
                )
            ),

        "n_features_scored":
            len(
                aurocs
            ),
    }


# ============================================================================
# 11. DIRECTION GEOMETRY
# ============================================================================

def normalize_rows(
    X,
):

    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-8,
    )

    return (
        X
        / norms
    )


def cosine_matrix(
    directions,
):

    normalized = normalize_rows(
        directions
    )

    return (
        normalized
        @ normalized.T
    )


def group_mean_abs(
    cosine_sim,
    indices,
):

    values = []

    for i in range(
        len(
            indices
        )
    ):

        for j in range(
            i + 1,
            len(
                indices
            )
        ):

            values.append(
                abs(
                    cosine_sim[
                        indices[i],
                        indices[j],
                    ]
                )
            )

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def pair_mean_abs(
    cosine_sim,
    pairs,
):

    values = [

        abs(
            cosine_sim[
                i,
                j,
            ]
        )

        for i, j in pairs
    ]

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


# ============================================================================
# 12. RIDGE-BASED FEATURE DIRECTIONS
# ============================================================================

def ridge_interference(
    model,
    X_probe,
    groups,
):

    """
    Fit:

        X -> Z

    with ridge regression.

    The fitted coefficient vector for each feature gives an estimated
    linear feature direction in bottleneck space.

    This is a useful global summary, but because the actual model is nonlinear,
    it should be interpreted alongside intervention-based measurements.
    """

    Z = get_bottleneck(
        model,
        X_probe,
    )

    ridge = Ridge(
        alpha=1.0
    )

    ridge.fit(
        X_probe,
        Z,
    )

    directions = (
        ridge.coef_.T
    )

    cosine_sim = cosine_matrix(
        directions
    )

    return {

        "ridge_independent_mean_abs_cos":
            group_mean_abs(
                cosine_sim,
                groups[
                    "independent"
                ],
            ),

        "ridge_positive_pairs_mean_abs_cos":
            pair_mean_abs(
                cosine_sim,
                groups[
                    "positive_pairs"
                ],
            ),

        "ridge_exclusive_pairs_mean_abs_cos":
            pair_mean_abs(
                cosine_sim,
                groups[
                    "exclusive_pairs"
                ],
            ),
    }


# ============================================================================
# 13. INTERVENTION-BASED FEATURE DIRECTIONS
# ============================================================================

def intervention_directions(
    model,
    X_baseline,
    n_baselines=N_INTERVENTION_BASELINES,
    seed=0,
):

    """
    For each feature i:

        1. Take real baseline examples.
        2. Force feature i = 1.
        3. Force feature i = 0.
        4. Compute the bottleneck difference.

    This measures what the model actually does when a feature is toggled.

    We retain:
        - the global mean direction
        - every contextual intervention vector

    The contextual vectors allow a more stringent analysis of interference.
    """

    rng = np.random.default_rng(
        seed
    )

    n = min(
        n_baselines,
        len(
            X_baseline
        ),
    )

    indices = rng.choice(
        len(
            X_baseline
        ),
        size=n,
        replace=False,
    )

    base = (
        X_baseline[
            indices
        ].copy()
    )

    global_directions = np.zeros(
        (
            N_FEATURES,
            model.hidden_dim,
        ),
        dtype=np.float32,
    )

    contextual_deltas = np.zeros(
        (
            N_FEATURES,
            n,
            model.hidden_dim,
        ),
        dtype=np.float32,
    )

    for i in range(
        N_FEATURES
    ):

        X_on = (
            base.copy()
        )

        X_off = (
            base.copy()
        )

        X_on[
            :,
            i
        ] = 1.0

        X_off[
            :,
            i
        ] = 0.0

        Z_on = get_bottleneck(
            model,
            X_on,
        )

        Z_off = get_bottleneck(
            model,
            X_off,
        )

        deltas = (
            Z_on
            - Z_off
        )

        contextual_deltas[
            i
        ] = deltas

        global_directions[
            i
        ] = deltas.mean(
            axis=0
        )

    return (
        global_directions,
        contextual_deltas,
    )


def intervention_interference(
    model,
    X_probe,
    groups,
):

    global_directions, contextual_deltas = (
        intervention_directions(
            model,
            X_probe,
        )
    )

    cosine_sim = cosine_matrix(
        global_directions
    )

    return {

        "interv_independent_mean_abs_cos":
            group_mean_abs(
                cosine_sim,
                groups[
                    "independent"
                ],
            ),

        "interv_positive_pairs_mean_abs_cos":
            pair_mean_abs(
                cosine_sim,
                groups[
                    "positive_pairs"
                ],
            ),

        "interv_exclusive_pairs_mean_abs_cos":
            pair_mean_abs(
                cosine_sim,
                groups[
                    "exclusive_pairs"
                ],
            ),
    }


# ============================================================================
# 14. RANDOM-DIRECTION NULL BASELINE
# ============================================================================

def random_direction_null(
    hidden_dim,
    n_features=N_FEATURES,
    n_repeats=1000,
    seed=0,
):

    """
    Estimate the expected absolute cosine similarity between random directions
    in the same dimensionality as the model bottleneck.

    This matters because even completely unrelated random directions have
    nonzero absolute cosine similarity.
    """

    rng = np.random.default_rng(
        seed
    )

    values = []

    for _ in range(
        n_repeats
    ):

        directions = rng.normal(
            size=(
                n_features,
                hidden_dim,
            )
        )

        cosine_sim = cosine_matrix(
            directions
        )

        values.append(
            group_mean_abs(
                cosine_sim,
                list(
                    range(
                        n_features
                    )
                ),
            )
        )

    return {

        "random_null_mean":
            float(
                np.mean(
                    values
                )
            ),

        "random_null_std":
            float(
                np.std(
                    values
                )
            ),
    }


# ============================================================================
# 15. ATTENTION MECHANISTIC STATISTICS
# ============================================================================

def mechanistic_stats(
    model,
    X,
):

    model.eval()

    with torch.no_grad():

        X_t = torch.tensor(
            X,
            dtype=torch.float32,
        )

        _, _, weights = model(
            X_t,
            return_attn=True,
        )

        W = (
            weights.numpy()
        )

    model.train()

    epsilon = 1e-12

    entropy = -(
        W
        * np.log(
            W
            + epsilon
        )
    ).sum(
        axis=-1
    ).mean()

    max_weight = (
        W.max(
            axis=-1
        ).mean()
    )

    effective_support = (
        1.0
        / (
            W ** 2
        ).sum(
            axis=-1
        )
    ).mean()

    fraction_exact_zero = (
        W
        == 0
    ).mean()

    return {

        "attention_entropy":
            float(
                entropy
            ),

        "max_attention_weight":
            float(
                max_weight
            ),

        "effective_support_size":
            float(
                effective_support
            ),

        "fraction_exact_zero_weights":
            float(
                fraction_exact_zero
            ),
    }


# ============================================================================
# 16. RUN ONE EXPERIMENTAL CELL
# ============================================================================

def run_single_condition(
    dataset_name,
    generator,
    groups,
    kernel_name,
    bottleneck_name,
    hidden_dim,
    seed,
):

    # --------------------------------------------------------
    # Independent data splits.
    # --------------------------------------------------------

    X_train = generator(
        N_TRAIN,
        seed=seed * 100 + 1,
    )

    X_test = generator(
        N_TEST,
        seed=seed * 100 + 2,
    )

    X_probe_train = generator(
        N_PROBE_TRAIN,
        seed=seed * 100 + 3,
    )

    X_probe_eval = generator(
        N_PROBE_EVAL,
        seed=seed * 100 + 4,
    )

    # --------------------------------------------------------
    # Train model.
    # --------------------------------------------------------

    model = train_model(
        X_train=X_train,
        hidden_dim=hidden_dim,
        kernel=kernel_name,
        seed=seed,
    )

    # --------------------------------------------------------
    # Reconstruction.
    # --------------------------------------------------------

    reconstruction = (
        evaluate_reconstruction(
            model,
            X_test,
        )
    )

    # --------------------------------------------------------
    # Linear probe recovery.
    # --------------------------------------------------------

    probes = probe_recovery(
        model,
        X_probe_train,
        X_probe_eval,
    )

    # --------------------------------------------------------
    # Ridge feature geometry.
    # --------------------------------------------------------

    ridge = ridge_interference(
        model,
        X_probe_train,
        groups,
    )

    # --------------------------------------------------------
    # Intervention feature geometry.
    # --------------------------------------------------------

    intervention = (
        intervention_interference(
            model,
            X_probe_train,
            groups,
        )
    )

    # --------------------------------------------------------
    # Attention statistics.
    # --------------------------------------------------------

    mechanics = mechanistic_stats(
        model,
        X_test,
    )

    # --------------------------------------------------------
    # Return one complete result row.
    # --------------------------------------------------------

    return {

        "dataset":
            dataset_name,

        "kernel":
            kernel_name,

        "bottleneck":
            bottleneck_name,

        "hidden_dim":
            hidden_dim,

        "seed":
            seed,

        **reconstruction,

        **probes,

        **ridge,

        **intervention,

        **mechanics,
    }


# ============================================================================
# 17. FULL FACTORIAL EXPERIMENT
# ============================================================================

def run_experiment():

    groups_A = {

        "independent":
            list(
                range(
                    N_FEATURES
                )
            ),

        "positive_pairs":
            [],

        "exclusive_pairs":
            [],
    }

    groups_B = {

        "independent":
            GROUP_INDEPENDENT,

        "positive_pairs":
            GROUP_POSITIVE_PAIRS,

        "exclusive_pairs":
            GROUP_EXCLUSIVE_PAIRS,
    }

    datasets = [

        (
            "A_independent",
            generate_dataset_A,
            groups_A,
        ),

        (
            "B_structured",
            generate_dataset_B,
            groups_B,
        ),
    ]

    results = []

    total_conditions = (
        len(
            datasets
        )
        * len(
            BOTTLENECKS
        )
        * len(
            ALL_KERNELS
        )
        * len(
            SEEDS
        )
    )

    completed = 0

    print("=" * 80)
    print(
        "RUNNING FULL EXPERIMENT"
    )
    print("=" * 80)

    print(
        f"\nTotal model fits: "
        f"{total_conditions}"
    )

    print(
        f"Formal 2×2 fits: "
        f"{len(datasets) * 2 * 2 * len(SEEDS)}"
    )

    print(
        f"Contrast fits: "
        f"{len(datasets) * 2 * 1 * len(SEEDS)}"
    )

    print()

    for (
        dataset_name,
        generator,
        groups,
    ) in datasets:

        for (
            bottleneck_name,
            hidden_dim,
        ) in BOTTLENECKS.items():

            for kernel_name in ALL_KERNELS:

                for seed in SEEDS:

                    row = run_single_condition(
                        dataset_name=dataset_name,
                        generator=generator,
                        groups=groups,
                        kernel_name=kernel_name,
                        bottleneck_name=bottleneck_name,
                        hidden_dim=hidden_dim,
                        seed=seed,
                    )

                    results.append(
                        row
                    )

                    completed += 1

                    print(
                        f"[{completed:3d}/"
                        f"{total_conditions}] "
                        f"{dataset_name:14s} | "
                        f"{kernel_name:8s} | "
                        f"{bottleneck_name:5s} | "
                        f"seed={seed} | "
                        f"BCE="
                        f"{row['recon_bce']:.4f} | "
                        f"AUROC="
                        f"{row['mean_auroc']:.3f} | "
                        f"ridge|cos|="
                        f"{row['ridge_independent_mean_abs_cos']:.3f}"
                    )

    return pd.DataFrame(
        results
    )


# ============================================================================
# 18. SUMMARY TABLES
# ============================================================================

def summarize_results(
    df,
):

    metric_columns = [

        "recon_bce",

        "mean_balanced_accuracy",

        "mean_auroc",

        "mean_average_precision",

        "mean_probe_balanced_accuracy",

        "ridge_independent_mean_abs_cos",

        "ridge_positive_pairs_mean_abs_cos",

        "ridge_exclusive_pairs_mean_abs_cos",

        "interv_independent_mean_abs_cos",

        "interv_positive_pairs_mean_abs_cos",

        "interv_exclusive_pairs_mean_abs_cos",

        "attention_entropy",

        "max_attention_weight",

        "effective_support_size",

        "fraction_exact_zero_weights",
    ]

    summary = (
        df
        .groupby(
            [
                "dataset",
                "kernel",
                "bottleneck",
                "hidden_dim",
            ]
        )[
            metric_columns
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    return summary


# ============================================================================
# 19. FORMAL 2×2 FACTORIAL ANALYSIS
# ============================================================================

def factorial_analysis(
    df,
):

    """
    Analyze only the formal 2×2 design:

        kernel:
            softmax vs bounded

        bottleneck:
            tight vs wide

    The key quantity is the interaction:

        (bounded_tight - softmax_tight)
        -
        (bounded_wide - softmax_wide)

    If this interaction is large, the effect of the kernel depends on
    bottleneck pressure.
    """

    factorial_df = df[
        df["kernel"].isin(
            FACTORIAL_KERNELS
        )
    ].copy()

    print("=" * 80)
    print(
        "FORMAL 2×2 FACTORIAL ANALYSIS"
    )
    print("=" * 80)

    print(
        "\nCell means:"
    )

    cell_means = (
        factorial_df
        .groupby(
            [
                "dataset",
                "kernel",
                "bottleneck",
            ]
        )[
            [
                "recon_bce",
                "mean_auroc",
                "ridge_independent_mean_abs_cos",
                "interv_independent_mean_abs_cos",
                "effective_support_size",
            ]
        ]
        .mean()
    )

    print(
        cell_means.to_string()
    )

    print(
        "\nKernel effects within each bottleneck:"
    )

    for dataset in (
        factorial_df[
            "dataset"
        ]
        .unique()
    ):

        print(
            f"\nDataset: "
            f"{dataset}"
        )

        for bottleneck in (
            BOTTLENECKS
        ):

            subset = factorial_df[
                (
                    factorial_df[
                        "dataset"
                    ]
                    == dataset
                )
                &
                (
                    factorial_df[
                        "bottleneck"
                    ]
                    == bottleneck
                )
            ]

            means = (
                subset
                .groupby(
                    "kernel"
                )[
                    [
                        "recon_bce",
                        "mean_auroc",
                        "ridge_independent_mean_abs_cos",
                        "interv_independent_mean_abs_cos",
                    ]
                ]
                .mean()
            )

            if (
                "softmax"
                in means.index
                and
                "bounded"
                in means.index
            ):

                print(
                    f"\n  Bottleneck: "
                    f"{bottleneck}"
                )

                for metric in [

                    "recon_bce",

                    "mean_auroc",

                    "ridge_independent_mean_abs_cos",

                    "interv_independent_mean_abs_cos",
                ]:

                    softmax_value = (
                        means.loc[
                            "softmax",
                            metric,
                        ]
                    )

                    bounded_value = (
                        means.loc[
                            "bounded",
                            metric,
                        ]
                    )

                    difference = (
                        bounded_value
                        - softmax_value
                    )

                    print(
                        f"    {metric}: "
                        f"bounded - softmax = "
                        f"{difference:+.6f}"
                    )

    print(
        "\nInterpretation:"
    )

    print(
        "The key question is whether the bounded-vs-softmax "
        "difference changes substantially between tight and wide "
        "bottlenecks. That is the kernel × bottleneck interaction."
    )

    return cell_means


# ============================================================================
# 20. RUN EVERYTHING
# ============================================================================

print_dataset_validation()

print("=" * 80)
print(
    "RANDOM-DIRECTION NULL BASELINES"
)
print("=" * 80)

for bottleneck_name, hidden_dim in BOTTLENECKS.items():

    null = random_direction_null(
        hidden_dim=hidden_dim,
        n_features=N_FEATURES,
        n_repeats=1000,
        seed=123,
    )

    print(
        f"{bottleneck_name:5s} "
        f"(hidden_dim={hidden_dim:2d}) | "
        f"random mean |cos| = "
        f"{null['random_null_mean']:.4f} "
        f"± "
        f"{null['random_null_std']:.4f}"
    )

print()

df_results = run_experiment()

print()

print("=" * 80)
print(
    "SUMMARY: MEAN ± STD ACROSS SEEDS"
)
print("=" * 80)

summary = summarize_results(
    df_results
)

print(
    summary.to_string()
)

print()

factorial_means = factorial_analysis(
    df_results
)

# Save raw per-run data.
df_results.to_csv(
    "results_2x2_raw.csv",
    index=False,
)

# Save grouped summary.
summary.to_csv(
    "results_2x2_summary.csv"
)

print()

print("=" * 80)
print(
    "FILES SAVED"
)
print("=" * 80)

print(
    "results_2x2_raw.csv"
)

print(
    "results_2x2_summary.csv"
)

print()

print(
    "EXPERIMENT COMPLETE."
)
