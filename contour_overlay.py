import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
from matplotlib.colors import TwoSlopeNorm
import analysis_functions as topo

def get_ranked_intervals(diagrams, dim=1):
    """
    Extract finite persistence intervals from one persistence diagram
    and sort them by persistence (longest first).
    """

    dgm = np.asarray(diagrams[dim])

    # Remove infinite intervals
    dgm = dgm[
        np.isfinite(dgm).all(axis=1)
    ]

    if len(dgm) == 0:
        return np.empty((0, 2))

    # Persistence = death - birth
    persistence = dgm[:, 1] - dgm[:, 0]

    # Longest first
    order = np.argsort(-persistence)

    return dgm[order]


def average_barcode(
    all_diagrams,
    dim=1,
    min_fraction=0.5,
):
    """
    Compute an average barcode across many persistence diagrams.

    Args:
        all_diagrams:
            List of persistence diagrams. Each element should be
            the output of ripser(...)[\"dgms\"].

        dim:
            Homology dimension.

        min_fraction:
            Minimum fraction of samples that must contain an interval
            at a given rank for that rank to be included.

    Returns:
        mean_intervals:
            Mean [birth, death] for each persistence rank.

        std_intervals:
            Standard deviation of [birth, death].

        counts:
            Number of samples contributing to each rank.
    """

    # Sort every sample's intervals by persistence
    ranked = [
        get_ranked_intervals(dgm, dim)
        for dgm in all_diagrams
    ]

    n_samples = len(ranked)

    if n_samples == 0:
        return (
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty(0)
        )

    # Maximum number of intervals in any sample
    max_intervals = max(
        len(x) for x in ranked
    )

    mean_intervals = []
    std_intervals = []
    counts = []

    for rank in range(max_intervals):

        # Collect interval at this rank from samples
        intervals = [
            x[rank]
            for x in ranked
            if len(x) > rank
        ]

        count = len(intervals)

        # Require enough samples to contribute
        if count < min_fraction * n_samples:
            continue

        intervals = np.asarray(intervals)

        mean = intervals.mean(axis=0)
        std = intervals.std(axis=0)

        mean_intervals.append(mean)
        std_intervals.append(std)
        counts.append(count)

    return (
        np.asarray(mean_intervals),
        np.asarray(std_intervals),
        np.asarray(counts)
    )

def plot_barcode_overlay(
    correct_diagrams,
    incorrect_diagrams,
    dim=1,
    figsize=(16, 7),
    alpha=0.65,
    linewidth=4,
    max_intervals=None,
    point_size=35,
):
    """
    Plot persistence diagrams and persistence barcodes side by side.

    LEFT:
        Persistence diagram
        x-axis = Birth
        y-axis = Death

    RIGHT:
        Persistence barcode
        x-axis = Filtration value
        y-axis = Persistence interval

    Correct = BLUE
    Incorrect = RED

    Args:
        correct_diagrams:
            Persistence diagram(s).

        incorrect_diagrams:
            Persistence diagram(s).

        dim:
            Homology dimension.

        figsize:
            Figure size.

        alpha:
            Transparency.

        linewidth:
            Barcode linewidth.

        max_intervals:
            Maximum number of intervals from each condition.
            Longest-persistence intervals are retained.

        point_size:
            Size of persistence diagram points.
    """

    # ================================================================
    # Get persistence diagrams
    # ================================================================

    correct_dgm = np.asarray(
        correct_diagrams[dim],
        dtype=float
    )

    incorrect_dgm = np.asarray(
        incorrect_diagrams[dim],
        dtype=float
    )

    # ================================================================
    # Remove infinite intervals
    # ================================================================

    correct_finite = correct_dgm[
        np.isfinite(correct_dgm).all(axis=1)
    ]

    incorrect_finite = incorrect_dgm[
        np.isfinite(incorrect_dgm).all(axis=1)
    ]

    # ================================================================
    # Sort by persistence
    # Longest persistence first
    # ================================================================

    if len(correct_finite) > 0:

        persistence = (
            correct_finite[:, 1]
            - correct_finite[:, 0]
        )

        correct_finite = correct_finite[
            np.argsort(-persistence)
        ]

    if len(incorrect_finite) > 0:

        persistence = (
            incorrect_finite[:, 1]
            - incorrect_finite[:, 0]
        )

        incorrect_finite = incorrect_finite[
            np.argsort(-persistence)
        ]

    # ================================================================
    # Limit intervals
    # ================================================================

    if max_intervals is not None:

        correct_finite = correct_finite[
            :max_intervals
        ]

        incorrect_finite = incorrect_finite[
            :max_intervals
        ]

    # ================================================================
    # Common axis limits
    # ================================================================

    all_points = []

    if len(correct_finite) > 0:
        all_points.append(correct_finite)

    if len(incorrect_finite) > 0:
        all_points.append(incorrect_finite)

    if len(all_points) > 0:

        all_points = np.vstack(all_points)

        xmin = all_points[:, 0].min()
        xmax = all_points[:, 1].max()

        ymin = all_points[:, 0].min()
        ymax = all_points[:, 1].max()

        x_range = xmax - xmin
        y_range = ymax - ymin

        x_pad = max(
            x_range * 0.08,
            0.01
        )

        y_pad = max(
            y_range * 0.08,
            0.01
        )

        xlim = (
            xmin - x_pad,
            xmax + x_pad
        )

        ylim = (
            ymin - y_pad,
            ymax + y_pad
        )

    else:

        xlim = (0, 1)
        ylim = (0, 1)

    # ================================================================
    # Create two-panel figure
    # ================================================================

    fig, (ax_diag, ax_bar) = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    # ================================================================
    # LEFT PANEL: PERSISTENCE DIAGRAM
    # ================================================================

    # ------------------------------------------------
    # Correct
    # ------------------------------------------------

    if len(correct_finite) > 0:

        ax_diag.scatter(
            correct_finite[:, 0],
            correct_finite[:, 1],
            color="#2563EB",
            s=point_size,
            alpha=0.75,
            marker="^",
            edgecolors="white",
            linewidths=0.6,
            label="Correct",
            zorder=5
        )

    # ------------------------------------------------
    # Incorrect
    # ------------------------------------------------

    if len(incorrect_finite) > 0:

        ax_diag.scatter(
            incorrect_finite[:, 0],
            incorrect_finite[:, 1],
            color="#DC2626",
            s=point_size,
            alpha=0.75,
            marker="o",
            edgecolors="white",
            linewidths=0.6,
            label="Incorrect",
            zorder=5
        )

    # ------------------------------------------------
    # Birth = Death
    # ------------------------------------------------

    diag_min = min(
        xlim[0],
        ylim[0]
    )

    diag_max = max(
        xlim[1],
        ylim[1]
    )

    ax_diag.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="Birth = Death",
        zorder=1
    )

    # ------------------------------------------------
    # Formatting
    # ------------------------------------------------

    ax_diag.set_xlim(xlim)
    ax_diag.set_ylim(ylim)

    ax_diag.set_xlabel(
        "Birth",
        fontsize=13
    )

    ax_diag.set_ylabel(
        "Death",
        fontsize=13
    )

    ax_diag.set_title(
        f"Persistence Diagram — H{dim}",
        fontsize=15,
        fontweight="bold"
    )

    ax_diag.grid(
        True,
        alpha=0.25
    )

    ax_diag.legend(
        loc="lower right",
        fontsize=10
    )

    # ================================================================
    # RIGHT PANEL: BARCODE
    # ================================================================

    n_correct = len(correct_finite)
    n_incorrect = len(incorrect_finite)

    max_y = max(
        n_correct,
        n_incorrect,
        1
    )

    # ------------------------------------------------
    # Give each condition its own vertical region
    # ------------------------------------------------

    correct_y = np.arange(
        n_correct
    )

    incorrect_y = np.arange(
        n_incorrect
    )

    # ------------------------------------------------
    # Correct
    # ------------------------------------------------

    for i, (birth, death) in enumerate(
        correct_finite
    ):

        y = correct_y[i]

        ax_bar.plot(
            [birth, death],
            [y, y],
            color="#2563EB",
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2
        )

        # Birth
        ax_bar.scatter(
            birth,
            y,
            color="#1E3A8A",
            s=25,
            zorder=3
        )

        # Death
        ax_bar.scatter(
            death,
            y,
            color="#1E3A8A",
            s=25,
            zorder=3
        )

    # ------------------------------------------------
    # Incorrect
    # ------------------------------------------------

    for i, (birth, death) in enumerate(
        incorrect_finite
    ):

        y = incorrect_y[i]

        ax_bar.plot(
            [birth, death],
            [y, y],
            color="#DC2626",
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=4
        )

        # Birth
        ax_bar.scatter(
            birth,
            y,
            color="#991B1B",
            s=25,
            zorder=5
        )

        # Death
        ax_bar.scatter(
            death,
            y,
            color="#991B1B",
            s=25,
            zorder=5
        )

    # ------------------------------------------------
    # Barcode legend
    # ------------------------------------------------

    ax_bar.plot(
        [],
        [],
        color="#2563EB",
        linewidth=linewidth,
        label="Correct"
    )

    ax_bar.plot(
        [],
        [],
        color="#DC2626",
        linewidth=linewidth,
        label="Incorrect"
    )

    # ------------------------------------------------
    # Formatting
    # ------------------------------------------------

    ax_bar.set_xlim(
        xlim
    )

    ax_bar.set_ylim(
        -1,
        max_y
    )

    ax_bar.set_xlabel(
        "Filtration value",
        fontsize=13
    )

    ax_bar.set_ylabel(
        "Persistence interval",
        fontsize=13
    )

    ax_bar.set_title(
        f"Persistence Barcode — H{dim}",
        fontsize=15,
        fontweight="bold"
    )

    ax_bar.legend(
        loc="upper right",
        fontsize=10
    )

    ax_bar.grid(
        axis="x",
        alpha=0.25
    )

    # ================================================================
    # Overall title
    # ================================================================

    fig.suptitle(
        f"Persistence Diagram and Barcode Overlay — H{dim}",
        fontsize=17,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    plt.show()


def sample_helper(model, dataset, num=100, avg=False):
    correct, incorrect = topo.rand_sample(model, dataset, n=num)
    
    if num == 1:
        correct_diagram = topo.parse_diagram_string(correct["diagrams"].iloc[0])
        incorrect_diagram = topo.parse_diagram_string(incorrect["diagrams"].iloc[0])

    else:
        correct_diagram = correct["diagrams"]
        incorrect_diagram = incorrect["diagrams"]

        correct_diagram = [topo.parse_diagram_string(x) for x in correct_diagram]
        incorrect_diagram = [topo.parse_diagram_string(x) for x in incorrect_diagram]
            
        if avg:
            # Compute average barcodes (returns mean, std, counts tuple)
            correct_diagram = average_barcode(correct_diagram, dim=0)
            incorrect_diagram = average_barcode(incorrect_diagram, dim=0)
    
    return correct_diagram, incorrect_diagram

# ==============================================================================
# KDE HELPER
# ==============================================================================

def compute_kde_2d(dgm, xlim, ylim, gridsize=150):

    finite = np.asarray(dgm)
    finite = finite[np.isfinite(finite).all(axis=1)]

    X, Y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], gridsize),
        np.linspace(ylim[0], ylim[1], gridsize)
    )

    if len(finite) < 3:
        return X, Y, np.zeros_like(X)

    x = finite[:, 0]
    y = finite[:, 1]

    try:
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)

        positions = np.vstack([
            X.ravel(),
            Y.ravel()
        ])

        Z = kernel(positions).reshape(X.shape)

    except (np.linalg.LinAlgError, ValueError):
        Z = np.zeros_like(X)

    return X, Y, Z


# ==============================================================================
# GET DIAGRAM
# ==============================================================================

def _get_diagram(diagrams, sample_idx=0, dim=1):
    """
    Extract a persistence diagram while supporting:

    1. Full Ripser-style structure:
           diagrams[sample_idx][dim]

    2. Dimension-specific structure:
           diagrams[sample_idx]

    3. A single persistence diagram:
           diagrams = [[birth, death], ...]
    """

    # ------------------------------------------------------------------
    # Case 1: diagrams is already a single N x 2 persistence diagram
    # ------------------------------------------------------------------

    try:
        candidate = np.asarray(diagrams, dtype=float)

        if candidate.ndim == 2 and candidate.shape[1] == 2:
            return candidate

    except (ValueError, TypeError):
        pass

    # ------------------------------------------------------------------
    # Get one sample WITHOUT converting the entire ragged collection
    # to a NumPy array.
    # ------------------------------------------------------------------

    try:
        sample = diagrams[sample_idx]
    except (IndexError, TypeError) as e:
        raise ValueError(
            f"Could not access sample {sample_idx} "
            f"from diagrams."
        ) from e

    # ------------------------------------------------------------------
    # Case 2: sample is already an N x 2 persistence diagram
    #
    # This is likely the structure of correct_dim1 / avg_correct_dim100
    # ------------------------------------------------------------------

    try:
        candidate = np.asarray(sample, dtype=float)

        if candidate.ndim == 2 and candidate.shape[1] == 2:
            return candidate

    except (ValueError, TypeError):
        pass

    # ------------------------------------------------------------------
    # Case 3: sample contains multiple homology dimensions:
    #
    # sample[0] = H0
    # sample[1] = H1
    # ...
    # ------------------------------------------------------------------

    try:
        candidate = np.asarray(sample[dim], dtype=float)

        if candidate.ndim == 2 and candidate.shape[1] == 2:
            return candidate

    except (ValueError, TypeError, IndexError):
        pass

    # ------------------------------------------------------------------
    # Nothing matched
    # ------------------------------------------------------------------

    raise ValueError(
        f"Could not extract persistence diagram.\n"
        f"sample_idx={sample_idx}, dim={dim}\n"
        f"Type of diagrams: {type(diagrams)}\n"
        f"Type of sample: {type(sample)}"
    )


# ==============================================================================
# CONTOUR OVERLAY
# ==============================================================================

def plot_contour_overlay(
    correct_diagrams,
    incorrect_diagrams,
    dim=1,
    sample_idx=0,
    figsize=(10, 8),
    contour_levels=8,
    point_size=20,
):
    """
    Persistence diagram contour overlay.

    IMPORTANT:
        Correct = BLUE
        Incorrect = RED

    x-axis = Birth
    y-axis = Death

    Supports both:

        full_dgms[sample][dim]

    and:

        dimension_specific_dgms[sample]
    """

    # ==========================================================================
    # Get diagrams
    # ==========================================================================

    dgm_correct = _get_diagram(
        correct_diagrams,
        sample_idx=sample_idx,
        dim=dim
    )

    dgm_incorrect = _get_diagram(
        incorrect_diagrams,
        sample_idx=sample_idx,
        dim=dim
    )

    # ==========================================================================
    # Remove infinite intervals
    # ==========================================================================

    finite_correct = dgm_correct[
        np.isfinite(dgm_correct).all(axis=1)
    ]

    finite_incorrect = dgm_incorrect[
        np.isfinite(dgm_incorrect).all(axis=1)
    ]

    # ==========================================================================
    # Axis limits
    # ==========================================================================

    all_points = []

    if len(finite_correct) > 0:
        all_points.append(finite_correct)

    if len(finite_incorrect) > 0:
        all_points.append(finite_incorrect)

    if all_points:

        all_points = np.vstack(all_points)

        xmin = all_points[:, 0].min()
        xmax = all_points[:, 0].max()

        ymin = all_points[:, 1].min()
        ymax = all_points[:, 1].max()

        x_range = xmax - xmin
        y_range = ymax - ymin

        x_pad = max(x_range * 0.10, 0.001)
        y_pad = max(y_range * 0.10, 0.001)

        xlim = (
            xmin - x_pad,
            xmax + x_pad
        )

        ylim = (
            ymin - y_pad,
            ymax + y_pad
        )

    else:

        xlim = (0, 1)
        ylim = (0, 1)

        print(
            f"⚠ No finite points found "
            f"for H{dim}, sample {sample_idx}"
        )

    # ==========================================================================
    # Figure
    # ==========================================================================

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ==========================================================================
    # CORRECT KDE
    # ==========================================================================

    if len(finite_correct) >= 3:

        X, Y, Z_correct = compute_kde_2d(
            finite_correct,
            xlim,
            ylim
        )

        if Z_correct.max() > 0:

            ax.contourf(
                X,
                Y,
                Z_correct,
                levels=contour_levels,
                cmap="Blues",
                alpha=0.45
            )

            ax.contour(
                X,
                Y,
                Z_correct,
                levels=contour_levels,
                colors="blue",
                linewidths=1,
                alpha=0.8
            )

    # ==========================================================================
    # INCORRECT KDE
    # ==========================================================================

    if len(finite_incorrect) >= 3:

        X, Y, Z_incorrect = compute_kde_2d(
            finite_incorrect,
            xlim,
            ylim
        )

        if Z_incorrect.max() > 0:

            ax.contourf(
                X,
                Y,
                Z_incorrect,
                levels=contour_levels,
                cmap="Reds",
                alpha=0.45
            )

            ax.contour(
                X,
                Y,
                Z_incorrect,
                levels=contour_levels,
                colors="red",
                linewidths=1,
                alpha=0.8
            )

    # ==========================================================================
    # CORRECT POINTS
    # ==========================================================================

    if len(finite_correct) > 0:

        ax.scatter(
            finite_correct[:, 0],
            finite_correct[:, 1],

            c="darkblue",

            s=point_size,

            alpha=0.7,

            marker="^",

            edgecolors="white",

            linewidths=0.5,

            label="Correct",

            zorder=5
        )

    # ==========================================================================
    # INCORRECT POINTS
    # ==========================================================================

    if len(finite_incorrect) > 0:

        ax.scatter(
            finite_incorrect[:, 0],
            finite_incorrect[:, 1],

            c="darkred",

            s=point_size,

            alpha=0.7,

            marker="o",

            edgecolors="white",

            linewidths=0.5,

            label="Incorrect",

            zorder=5
        )

    # ==========================================================================
    # Birth = Death
    # ==========================================================================

    diag_min = min(
        xlim[0],
        ylim[0]
    )

    diag_max = max(
        xlim[1],
        ylim[1]
    )

    ax.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],

        color="gray",

        linestyle="--",

        linewidth=1.5,

        alpha=0.6,

        label="Birth = Death",

        zorder=4
    )

    # ==========================================================================
    # Formatting
    # ==========================================================================

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(
        "Birth",
        fontsize=12
    )

    ax.set_ylabel(
        "Death",
        fontsize=12
    )

    ax.set_title(
        f"Contour Overlay - H{dim} - Sample {sample_idx}",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(
        loc="lower right",
        fontsize=10
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

def plot_mean_persistence_landscape(
    correct_diagrams,
    incorrect_diagrams,
    dim=1,
    num_layers=5,
    num_points=500,
    figsize=(12, 7),
    show_std=True,
):
    """
    Plot mean persistence landscapes for:

        Correct = BLUE
        Incorrect = RED

    Supports both:

        1. Full Ripser structure:
           diagrams[sample][dim]

        2. Already-selected dimension:
           diagrams[sample]

    x-axis:
        Filtration value

    y-axis:
        Persistence landscape lambda_k
    """

    # ================================================================
    # Helper: convert one persistence diagram into a landscape
    # ================================================================

    def compute_landscape(dgm, x_grid, num_layers):

        # Explicitly force numeric N x 2 array
        dgm = np.asarray(dgm, dtype=float)

        # Safety check
        if dgm.ndim != 2 or dgm.shape[1] != 2:
            raise ValueError(
                f"Expected persistence diagram with shape (N, 2), "
                f"got {dgm.shape}"
            )

        # Remove infinite intervals
        dgm = dgm[
            np.isfinite(dgm).all(axis=1)
        ]

        if len(dgm) == 0:
            return np.zeros(
                (num_layers, len(x_grid))
            )

        # ============================================================
        # Compute tent functions
        # ============================================================

        tents = []

        for birth, death in dgm:

            if death <= birth:
                continue

            values = np.maximum(
                0,
                np.minimum(
                    x_grid - birth,
                    death - x_grid
                )
            )

            tents.append(values)

        if len(tents) == 0:
            return np.zeros(
                (num_layers, len(x_grid))
            )

        tents = np.asarray(
            tents,
            dtype=float
        )

        # ============================================================
        # Sort tents at each x
        # ============================================================

        tents = np.sort(
            tents,
            axis=0
        )[::-1]

        # ============================================================
        # Keep requested number of layers
        # ============================================================

        if tents.shape[0] >= num_layers:

            landscape = tents[:num_layers]

        else:

            landscape = np.zeros(
                (num_layers, len(x_grid))
            )

            landscape[:tents.shape[0]] = tents

        return landscape

    # ================================================================
    # Get diagrams
    # ================================================================
    #
    # IMPORTANT:
    # _get_diagram handles both:
    #
    #   diagrams[sample][dim]
    #
    # and
    #
    #   diagrams[sample]
    #
    # ================================================================

    correct_dgms = []

    for i in range(len(correct_diagrams)):

        dgm = _get_diagram(
            correct_diagrams,
            sample_idx=i,
            dim=dim
        )

        dgm = np.asarray(
            dgm,
            dtype=float
        )

        correct_dgms.append(dgm)

    incorrect_dgms = []

    for i in range(len(incorrect_diagrams)):

        dgm = _get_diagram(
            incorrect_diagrams,
            sample_idx=i,
            dim=dim
        )

        dgm = np.asarray(
            dgm,
            dtype=float
        )

        incorrect_dgms.append(dgm)

    # ================================================================
    # Determine common filtration range
    # ================================================================

    all_points = []

    for dgm in correct_dgms + incorrect_dgms:

        dgm = np.asarray(
            dgm,
            dtype=float
        )

        finite = dgm[
            np.isfinite(dgm).all(axis=1)
        ]

        if len(finite) > 0:
            all_points.append(finite)

    if len(all_points) == 0:

        raise ValueError(
            f"No finite persistence points found for H{dim}."
        )

    # ================================================================
    # Common range
    # ================================================================

    all_points = np.vstack(
        all_points
    )

    xmin = all_points[:, 0].min()
    xmax = all_points[:, 1].max()

    padding = max(
        (xmax - xmin) * 0.05,
        1e-6
    )

    xmin -= padding
    xmax += padding

    x_grid = np.linspace(
        xmin,
        xmax,
        num_points
    )

    # ================================================================
    # Compute landscapes
    # ================================================================

    correct_landscapes = np.asarray(
        [
            compute_landscape(
                dgm,
                x_grid,
                num_layers
            )
            for dgm in correct_dgms
        ],
        dtype=float
    )

    incorrect_landscapes = np.asarray(
        [
            compute_landscape(
                dgm,
                x_grid,
                num_layers
            )
            for dgm in incorrect_dgms
        ],
        dtype=float
    )

    # ================================================================
    # Mean / standard deviation
    # ================================================================

    correct_mean = correct_landscapes.mean(
        axis=0
    )

    incorrect_mean = incorrect_landscapes.mean(
        axis=0
    )

    correct_std = correct_landscapes.std(
        axis=0
    )

    incorrect_std = incorrect_landscapes.std(
        axis=0
    )

    # ================================================================
    # Plot
    # ================================================================

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # Better blue gradient
    colors_correct = [
        "#00008B",  # dark blue
        "#1D4ED8",
        "#3B82F6",
        "#60A5FA",
        "#93C5FD",
    ]

    # Better red gradient
    colors_incorrect = [
        "#8B0000",  # dark red
        "#B22222",
        "#DC2626",
        "#EF4444",
        "#F87171",
    ]

    # ================================================================
    # Plot landscape layers
    # ================================================================

    for k in range(num_layers):

        color_correct = colors_correct[
            min(k, len(colors_correct) - 1)
        ]

        color_incorrect = colors_incorrect[
            min(k, len(colors_incorrect) - 1)
        ]

        # ------------------------------------------------------------
        # Correct
        # ------------------------------------------------------------

        ax.plot(
            x_grid,
            correct_mean[k],
            color=color_correct,
            linewidth=2,
            alpha=0.9,
            label=(
                "Correct"
                if k == 0
                else None
            )
        )

        # ------------------------------------------------------------
        # Incorrect
        # ------------------------------------------------------------

        ax.plot(
            x_grid,
            incorrect_mean[k],
            color=color_incorrect,
            linewidth=2,
            alpha=0.9,
            label=(
                "Incorrect"
                if k == 0
                else None
            )
        )

    # ================================================================
    # Standard deviation
    # ================================================================

    if show_std:

        # Correct
        ax.fill_between(
            x_grid,

            np.maximum(
                0,
                correct_mean[0] - correct_std[0]
            ),

            correct_mean[0] + correct_std[0],

            color="#3B82F6",

            alpha=0.12,

            label="Correct ±1 SD"
        )

        # Incorrect
        ax.fill_between(
            x_grid,

            np.maximum(
                0,
                incorrect_mean[0] - incorrect_std[0]
            ),

            incorrect_mean[0] + incorrect_std[0],

            color="#EF4444",

            alpha=0.12,

            label="Incorrect ±1 SD"
        )

    # ================================================================
    # Formatting
    # ================================================================

    ax.set_xlabel(
        "Filtration value",
        fontsize=13
    )

    ax.set_ylabel(
        r"Persistence landscape $\lambda_k$",
        fontsize=13
    )

    ax.set_title(
        f"Mean Persistence Landscape — H{dim}",
        fontsize=15,
        fontweight="bold"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.3
    )

    ax.grid(
        True,
        alpha=0.2
    )

    ax.legend(
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

    # ================================================================
    # Return
    # ================================================================

    return {
        "x": x_grid,

        "correct_mean": correct_mean,
        "correct_std": correct_std,

        "incorrect_mean": incorrect_mean,
        "incorrect_std": incorrect_std,

        "correct_landscapes": correct_landscapes,
        "incorrect_landscapes": incorrect_landscapes,
    }


def plot_persistence_density_difference(
    correct_diagrams,
    incorrect_diagrams,
    dim=1,
    sample_indices=None,
    figsize=(18, 6),
    gridsize=150,
    contour_levels=10,
    bandwidth=None,
):
    """
    Compare persistence-diagram density between correct and incorrect samples.

    Panels:
        1. Correct density
        2. Incorrect density
        3. Correct - Incorrect density difference

    x-axis = Birth
    y-axis = Death

    Blue  = Correct
    Red   = Incorrect

    The difference panel uses:
        red  -> more incorrect density
        blue -> more correct density
        white -> similar density

    Parameters
    ----------
    correct_diagrams : list
        Persistence diagrams.

    incorrect_diagrams : list
        Persistence diagrams.

    dim : int
        Homology dimension.

    sample_indices : list or None
        Samples to include. None = all.

    bandwidth : float or None
        KDE bandwidth adjustment.
    """

    # ================================================================
    # Extract diagrams
    # ================================================================

    def get_dgm(dgm):

        arr = np.asarray(dgm, dtype=float)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected persistence diagram with shape (N,2), "
                f"got {arr.shape}"
            )

        return arr[
            np.isfinite(arr).all(axis=1)
        ]

    correct_points = []
    incorrect_points = []

    if sample_indices is None:

        correct_indices = range(len(correct_diagrams))
        incorrect_indices = range(len(incorrect_diagrams))

    else:

        correct_indices = sample_indices
        incorrect_indices = sample_indices

    # ------------------------------------------------
    # Collect correct
    # ------------------------------------------------

    for i in correct_indices:

        try:
            dgm = correct_diagrams[i]

            # Handle full dgms: [H0, H1, ...]
            if isinstance(dgm, (list, tuple)):

                dgm = dgm[dim]

            else:

                arr = np.asarray(dgm)

                if arr.ndim == 3:
                    dgm = arr[dim]

            dgm = get_dgm(dgm)

            if len(dgm) > 0:
                correct_points.append(dgm)

        except Exception:
            continue

    # ------------------------------------------------
    # Collect incorrect
    # ------------------------------------------------

    for i in incorrect_indices:

        try:
            dgm = incorrect_diagrams[i]

            if isinstance(dgm, (list, tuple)):

                dgm = dgm[dim]

            else:

                arr = np.asarray(dgm)

                if arr.ndim == 3:
                    dgm = arr[dim]

            dgm = get_dgm(dgm)

            if len(dgm) > 0:
                incorrect_points.append(dgm)

        except Exception:
            continue

    if not correct_points and not incorrect_points:

        raise ValueError(
            "No finite persistence points found."
        )

    # ================================================================
    # Stack points
    # ================================================================

    correct_points = (
        np.vstack(correct_points)
        if correct_points
        else np.empty((0, 2))
    )

    incorrect_points = (
        np.vstack(incorrect_points)
        if incorrect_points
        else np.empty((0, 2))
    )

    all_points = np.vstack([
        x for x in [
            correct_points,
            incorrect_points
        ]
        if len(x) > 0
    ])

    # ================================================================
    # Axis limits
    # ================================================================

    xmin = all_points[:, 0].min()
    xmax = all_points[:, 0].max()

    ymin = all_points[:, 1].min()
    ymax = all_points[:, 1].max()

    xpad = max(
        (xmax - xmin) * 0.08,
        1e-6
    )

    ypad = max(
        (ymax - ymin) * 0.08,
        1e-6
    )

    xlim = (
        xmin - xpad,
        xmax + xpad
    )

    ylim = (
        ymin - ypad,
        ymax + ypad
    )

    # ================================================================
    # Grid
    # ================================================================

    x = np.linspace(
        xlim[0],
        xlim[1],
        gridsize
    )

    y = np.linspace(
        ylim[0],
        ylim[1],
        gridsize
    )

    X, Y = np.meshgrid(x, y)

    positions = np.vstack([
        X.ravel(),
        Y.ravel()
    ])

    # ================================================================
    # KDE helper
    # ================================================================

    def kde(points):

        if len(points) < 3:

            return np.zeros_like(X)

        try:

            values = points.T

            kernel = gaussian_kde(
                values,
                bw_method=bandwidth
            )

            Z = kernel(
                positions
            ).reshape(X.shape)

            return Z

        except (
            np.linalg.LinAlgError,
            ValueError
        ):

            return np.zeros_like(X)

    # ================================================================
    # Compute KDEs
    # ================================================================

    Z_correct = kde(
        correct_points
    )

    Z_incorrect = kde(
        incorrect_points
    )

    # ================================================================
    # Difference
    # ================================================================

    Z_difference = ( Z_correct - Z_incorrect)

    # max_difference = np.max(np.abs(Z_difference))

    # ================================================================
    # Plot
    # ================================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize
    )

    ax1, ax2, ax3 = axes

    # ------------------------------------------------
    # Correct
    # ------------------------------------------------

    if Z_correct.max() > 0:

        ax1.contourf(
            X,
            Y,
            Z_correct,
            levels=contour_levels,
            cmap="Blues",
            alpha=0.75
        )

        ax1.contour(
            X,
            Y,
            Z_correct,
            levels=contour_levels,
            colors="blue",
            linewidths=0.8,
            alpha=0.7
        )

    ax1.scatter(
        correct_points[:, 0],
        correct_points[:, 1],
        color="darkblue",
        s=8,
        alpha=0.25
    )

    ax1.set_title(
        "Correct",
        fontweight="bold"
    )

    # ------------------------------------------------
    # Incorrect
    # ------------------------------------------------

    if Z_incorrect.max() > 0:

        ax2.contourf(
            X,
            Y,
            Z_incorrect,
            levels=contour_levels,
            cmap="Reds",
            alpha=0.75
        )

        ax2.contour(
            X,
            Y,
            Z_incorrect,
            levels=contour_levels,
            colors="red",
            linewidths=0.8,
            alpha=0.7
        )

    ax2.scatter(
        incorrect_points[:, 0],
        incorrect_points[:, 1],
        color="darkred",
        s=8,
        alpha=0.25
    )

    ax2.set_title(
        "Incorrect",
        fontweight="bold"
    )

    # ------------------------------------------------
    # Difference normalization
    # ------------------------------------------------

    difference = Z_correct - Z_incorrect

    max_difference = float(
        np.nanmax(np.abs(difference))
    )

    if not np.isfinite(max_difference):
        max_difference = 0.0

    if max_difference > 1e-12:

        norm = TwoSlopeNorm(
            vmin=-max_difference,
            vcenter=0.0,
            vmax=max_difference
        )

    else:

        # Correct and incorrect densities are identical
        # or numerically indistinguishable.
        norm = None


    # im = ax3.contourf(
    #     X,
    #     Y,
    #     Z_difference,
    #     levels=contour_levels * 2,
    #     cmap="RdBu_r",
    #     norm=norm,
    #     alpha=0.9
    # )
    if norm is not None:

        im = ax3.contourf(
            X,
            Y,
            difference,
            levels=contour_levels,
            cmap="RdBu_r",
            norm=norm
        )

    else:

        # No measurable difference
        im = ax3.contourf(
            X,
            Y,
            difference,
            levels=3,
            cmap="RdBu_r",
            vmin=-1e-12,
            vmax=1e-12
        )


    ax3.contour(
        X,
        Y,
        Z_difference,
        levels=contour_levels * 2,
        colors="black",
        linewidths=0.4,
        alpha=0.25
    )

    ax3.set_title(
        "Density Difference\nCorrect − Inorrect",
        fontweight="bold"
    )

    # ------------------------------------------------
    # Diagonal on all plots
    # ------------------------------------------------

    for ax in axes:

        lo = min(
            xlim[0],
            ylim[0]
        )

        hi = max(
            xlim[1],
            ylim[1]
        )

        ax.plot(
            [lo, hi],
            [lo, hi],
            "--",
            color="black",
            alpha=0.35,
            linewidth=1
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel(
            "Birth",
            fontsize=12
        )

        ax.set_ylabel(
            "Death",
            fontsize=12
        )

        ax.grid(
            alpha=0.2
        )

    # ------------------------------------------------
    # Colorbar
    # ------------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=ax3,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        "Incorrect density − Correct density",
        fontsize=11
    )

    fig.suptitle(
        f"Persistence Density Comparison — H{dim}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()

    return {
        "X": X,
        "Y": Y,
        "correct_density": Z_correct,
        "incorrect_density": Z_incorrect,
        "difference": Z_difference,
    }

def plot_h0_component_curves(
    correct_diagrams,
    incorrect_diagrams,
    figsize=(12, 7),
    num_points=500,
    show_std=True,
):
    """
    Plot H0 persistence component curves for Correct vs Incorrect.

    Handles both:

        full diagrams:
            [H0, H1, ...]

    and:

        H0-only diagrams:
            [[birth, death], ...]

    Correct   = blue
    Incorrect = red

    For H0, each finite interval is represented by its persistence:

        persistence = death - birth

    The curves show the distribution of H0 persistence values
    across samples.
    """

    # ================================================================
    # Extract H0 robustly
    # ================================================================

    def extract_h0(dgm):

        # ------------------------------------------------------------
        # First try to interpret as a numeric Nx2 diagram
        # ------------------------------------------------------------

        try:

            arr = np.asarray(
                dgm,
                dtype=float
            )

            if (
                arr.ndim == 2
                and arr.shape[1] == 2
            ):
                return arr

        except (ValueError, TypeError):
            pass

        # ------------------------------------------------------------
        # Otherwise assume full persistence diagram:
        #
        # dgm = [H0, H1, ...]
        # ------------------------------------------------------------

        try:

            h0 = np.asarray(
                dgm[0],
                dtype=float
            )

            if (
                h0.ndim == 2
                and h0.shape[1] == 2
            ):
                return h0

        except (ValueError, TypeError, IndexError):
            pass

        raise ValueError(
            "Could not extract H0 persistence diagram.\n"
            f"Input type: {type(dgm)}"
        )

    # ================================================================
    # Extract all H0 diagrams
    # ================================================================

    correct_h0 = []

    for dgm in correct_diagrams:

        h0 = extract_h0(dgm)

        h0 = h0[
            np.isfinite(h0).all(axis=1)
        ]

        if len(h0) > 0:
            correct_h0.append(h0)

    incorrect_h0 = []

    for dgm in incorrect_diagrams:

        h0 = extract_h0(dgm)

        h0 = h0[
            np.isfinite(h0).all(axis=1)
        ]

        if len(h0) > 0:
            incorrect_h0.append(h0)

    if len(correct_h0) == 0:
        raise ValueError(
            "No finite H0 diagrams found for Correct."
        )

    if len(incorrect_h0) == 0:
        raise ValueError(
            "No finite H0 diagrams found for Incorrect."
        )

    # ================================================================
    # Convert intervals to persistence values
    # ================================================================

    correct_persistence = [
        h0[:, 1] - h0[:, 0]
        for h0 in correct_h0
    ]

    incorrect_persistence = [
        h0[:, 1] - h0[:, 0]
        for h0 in incorrect_h0
    ]

    # ================================================================
    # Common persistence range
    # ================================================================

    all_persistence = np.concatenate(
        correct_persistence +
        incorrect_persistence
    )

    all_persistence = all_persistence[
        np.isfinite(all_persistence)
    ]

    if len(all_persistence) == 0:
        raise ValueError(
            "No finite H0 persistence values found."
        )

    xmin = all_persistence.min()
    xmax = all_persistence.max()

    padding = max(
        (xmax - xmin) * 0.05,
        1e-6
    )

    xmin -= padding
    xmax += padding

    x_grid = np.linspace(
        xmin,
        xmax,
        num_points
    )

    # ================================================================
    # Build empirical survival curves
    #
    # At each persistence value x:
    #
    # fraction of components with persistence >= x
    # ================================================================

    def component_curve(persistence):

        persistence = np.sort(
            persistence
        )

        curve = np.array([
            np.mean(
                persistence >= x
            )
            for x in x_grid
        ])

        return curve

    # ================================================================
    # Compute curves per sample
    # ================================================================

    correct_curves = np.asarray([
        component_curve(p)
        for p in correct_persistence
    ])

    incorrect_curves = np.asarray([
        component_curve(p)
        for p in incorrect_persistence
    ])

    # ================================================================
    # Mean and standard deviation
    # ================================================================

    correct_mean = correct_curves.mean(
        axis=0
    )

    incorrect_mean = incorrect_curves.mean(
        axis=0
    )

    correct_std = correct_curves.std(
        axis=0
    )

    incorrect_std = incorrect_curves.std(
        axis=0
    )

    # ================================================================
    # Plot
    # ================================================================

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ------------------------------------------------
    # Correct
    # ------------------------------------------------

    ax.plot(
        x_grid,
        correct_mean,
        color="darkblue",
        linewidth=3,
        label="Correct"
    )

    # ------------------------------------------------
    # Incorrect
    # ------------------------------------------------

    ax.plot(
        x_grid,
        incorrect_mean,
        color="darkred",
        linewidth=3,
        label="Incorrect"
    )

    # ================================================================
    # Standard deviation
    # ================================================================

    if show_std:

        ax.fill_between(
            x_grid,
            np.maximum(
                0,
                correct_mean - correct_std
            ),
            correct_mean + correct_std,
            color="blue",
            alpha=0.15,
            label="Correct ±1 SD"
        )

        ax.fill_between(
            x_grid,
            np.maximum(
                0,
                incorrect_mean - incorrect_std
            ),
            incorrect_mean + incorrect_std,
            color="red",
            alpha=0.15,
            label="Incorrect ±1 SD"
        )

    # ================================================================
    # Formatting
    # ================================================================

    ax.set_xlabel(
        "H0 persistence (death − birth)",
        fontsize=13
    )

    ax.set_ylabel(
        "Fraction of components with persistence ≥ x",
        fontsize=13
    )

    ax.set_title(
        "H0 Persistence Component Curves",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

    # ================================================================
    # Return results
    # ================================================================

    return {
        "x": x_grid,

        "correct_mean": correct_mean,
        "correct_std": correct_std,

        "incorrect_mean": incorrect_mean,
        "incorrect_std": incorrect_std,

        "correct_curves": correct_curves,
        "incorrect_curves": incorrect_curves,

        "correct_h0": correct_h0,
        "incorrect_h0": incorrect_h0,

        "correct_persistence": correct_persistence,
        "incorrect_persistence": incorrect_persistence,
    }

# helper for persistence
def extract_diagram(dgm, dim=0):
    try:
        arr = np.asarray(dgm, dtype=float)

        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr

    except (ValueError, TypeError):
        pass

    try:
        arr = np.asarray(
            dgm[dim],
            dtype=float
        )

        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr

    except (ValueError, TypeError, IndexError):
        pass

    raise ValueError(
        f"Could not extract H{dim} persistence diagram.\n"
        f"Input type: {type(dgm)}"
    )

# extract persistence variables
def get_persistence(dgm):

    arr = extract_diagram(dgm)

    # remove infinite intervals
    arr = arr[
        np.isfinite(arr).all(axis=1)
    ]

    if len(arr) == 0:
        return np.array([], dtype=float)

    persistence = (
        arr[:, 1] - arr[:, 0]
    )

    # Remove invalid / zero persistence
    persistence = persistence[
        np.isfinite(persistence)
    ]

    persistence = persistence[
        persistence > 0
    ]

    return persistence

def plot_persistence_heatmap(
    correct_diagrams,
    incorrect_diagrams,
    dim=0,
    num_bins=50,
    figsize=(12, 7),
    max_persistence=None,
):
    """
    Plot persistence-value distributions as heatmaps.

    Correct   = blue
    Incorrect = red

    Handles both:

        full diagrams:
            diagrams[sample][dim]

    and:

        dimension-specific diagrams:
            diagrams[sample]

    Persistence is:

        death - birth

    The heatmap shows the distribution of persistence values
    across samples.
    """

    # get persistence values for each sample
    correct_persistence = [
        get_persistence(dgm)
        for dgm in correct_diagrams
    ]

    incorrect_persistence = [
        get_persistence(dgm)
        for dgm in incorrect_diagrams
    ]

    # Remove empty samples
    correct_persistence = [
        p for p in correct_persistence
        if len(p) > 0
    ]

    incorrect_persistence = [
        p for p in incorrect_persistence
        if len(p) > 0
    ]

    if (
        len(correct_persistence) == 0
        and len(incorrect_persistence) == 0
    ):
        raise ValueError(
            f"No finite positive persistence values "
            f"found for H{dim}."
        )

    # ================================================================
    # Determine persistence range
    # ================================================================

    all_values = []

    if correct_persistence:
        all_values.extend(correct_persistence)

    if incorrect_persistence:
        all_values.extend(incorrect_persistence)

    all_values = np.concatenate(all_values)

    if max_persistence is None:
        max_persistence = all_values.max()

    min_persistence = 0.0

    # Small padding
    if max_persistence <= min_persistence:
        max_persistence = 1.0

    # ================================================================
    # Create common bins
    # ================================================================

    bins = np.linspace(
        min_persistence,
        max_persistence,
        num_bins + 1
    )

    bin_centers = (
        bins[:-1] + bins[1:]
    ) / 2

    # ================================================================
    # Build sample × persistence-bin matrices
    # ================================================================

    def make_histograms(persistence_list):

        if len(persistence_list) == 0:
            return np.zeros(
                (1, num_bins)
            )

        histograms = []

        for persistence in persistence_list:

            persistence = persistence[
                persistence <= max_persistence
            ]

            hist, _ = np.histogram(
                persistence,
                bins=bins
            )

            # Normalize so samples with different
            # numbers of components are comparable
            if hist.sum() > 0:
                hist = hist / hist.sum()

            histograms.append(hist)

        return np.asarray(histograms)

    correct_hist = make_histograms(
        correct_persistence
    )

    incorrect_hist = make_histograms(
        incorrect_persistence
    )

    # ================================================================
    # Plot
    # ================================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True
    )

    # ================================================================
    # Correct
    # ================================================================

    im1 = axes[0].imshow(
        correct_hist,
        aspect="auto",
        origin="lower",
        extent=[
            min_persistence,
            max_persistence,
            0,
            correct_hist.shape[0]
        ],
        cmap="Blues",
        interpolation="nearest"
    )

    axes[0].set_title(
        "Correct",
        fontsize=14,
        fontweight="bold"
    )

    axes[0].set_xlabel(
        "Persistence (death − birth)",
        fontsize=12
    )

    axes[0].set_ylabel(
        "Sample",
        fontsize=12
    )

    # ================================================================
    # Incorrect
    # ================================================================

    im2 = axes[1].imshow(
        incorrect_hist,
        aspect="auto",
        origin="lower",
        extent=[
            min_persistence,
            max_persistence,
            0,
            incorrect_hist.shape[0]
        ],
        cmap="Reds",
        interpolation="nearest"
    )

    axes[1].set_title(
        "Incorrect",
        fontsize=14,
        fontweight="bold"
    )

    axes[1].set_xlabel(
        "Persistence (death − birth)",
        fontsize=12
    )

    # ================================================================
    # Colorbars
    # ================================================================

    cbar1 = fig.colorbar(
        im1,
        ax=axes[0],
        fraction=0.046,
        pad=0.04
    )

    cbar1.set_label(
        "Fraction of intervals",
        fontsize=10
    )

    cbar2 = fig.colorbar(
        im2,
        ax=axes[1],
        fraction=0.046,
        pad=0.04
    )

    cbar2.set_label(
        "Fraction of intervals",
        fontsize=10
    )

    # ================================================================
    # Formatting
    # ================================================================

    fig.suptitle(
        f"Persistence Distribution Heatmap — H{dim}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

    # ================================================================
    # Return data
    # ================================================================

    return {
        "bins": bins,
        "bin_centers": bin_centers,

        "correct_hist": correct_hist,
        "incorrect_hist": incorrect_hist,

        "correct_persistence": correct_persistence,
        "incorrect_persistence": incorrect_persistence,
    }

# ==============================================================================
# HELPER: EXTRACT FINITE PERSISTENCE DIAGRAM
# ==============================================================================

def _extract_finite_diagram(diagrams, dim=0, sample_idx=0):
    """
    Extract one persistence diagram robustly.

    Supports:
        diagrams[sample][dim]
        diagrams[sample]
        diagrams = single N x 2 diagram
    """

    # ----------------------------------------------------------
    # First try direct numeric conversion
    # ----------------------------------------------------------

    try:
        arr = np.asarray(diagrams, dtype=float)

        # Single N x 2 diagram
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr

    except (ValueError, TypeError):
        pass

    # ----------------------------------------------------------
    # Sample
    # ----------------------------------------------------------

    try:
        sample = diagrams[sample_idx]
    except Exception:
        raise ValueError(
            f"Could not access sample {sample_idx}"
        )

    # ----------------------------------------------------------
    # Try sample[dim]
    # ----------------------------------------------------------

    try:
        candidate = np.asarray(
            sample[dim],
            dtype=float
        )

        if candidate.ndim == 2 and candidate.shape[1] == 2:
            return candidate

    except (ValueError, TypeError, IndexError):
        pass

    # ----------------------------------------------------------
    # Try sample itself
    # ----------------------------------------------------------

    try:
        candidate = np.asarray(
            sample,
            dtype=float
        )

        if candidate.ndim == 2 and candidate.shape[1] == 2:
            return candidate

    except (ValueError, TypeError):
        pass

    raise ValueError(
        f"Could not extract persistence diagram.\n"
        f"sample_idx={sample_idx}\n"
        f"dim={dim}\n"
        f"sample type={type(sample)}"
    )


# ==============================================================================
# COLLECT ALL DIAGRAMS
# ==============================================================================

def _collect_diagrams(diagrams, dim=0):
    """
    Convert a collection of persistence diagrams into
    a list of finite N x 2 arrays.
    """

    # Check whether this is already one diagram
    try:
        arr = np.asarray(diagrams, dtype=float)

        if arr.ndim == 2 and arr.shape[1] == 2:
            finite = arr[
                np.isfinite(arr).all(axis=1)
            ]

            return [finite]

    except (ValueError, TypeError):
        pass

    # Otherwise assume collection
    result = []

    for i in range(len(diagrams)):

        dgm = _extract_finite_diagram(
            diagrams,
            dim=dim,
            sample_idx=i
        )

        finite = dgm[
            np.isfinite(dgm).all(axis=1)
        ]

        result.append(finite)

    return result

# birth persistence overlay
def plot_birth_persistence_overlay(
    correct_diagrams,
    incorrect_diagrams,
    dim=0,
    figsize=(11, 8),
    point_size=28,
    alpha=0.45,
    kde=True,
    contour_levels=6,
):
    """
    Plot persistence diagrams in Birth × Persistence coordinates.

    Correct:
        blue

    Incorrect:
        red

    x-axis:
        Birth

    y-axis:
        Persistence = Death - Birth
    """

    correct_dgms = _collect_diagrams(
        correct_diagrams,
        dim=dim
    )

    incorrect_dgms = _collect_diagrams(
        incorrect_diagrams,
        dim=dim
    )

    # ----------------------------------------------------------
    # Combine all points
    # ----------------------------------------------------------

    correct_points = (
        np.vstack(correct_dgms)
        if any(len(x) > 0 for x in correct_dgms)
        else np.empty((0, 2))
    )

    incorrect_points = (
        np.vstack(incorrect_dgms)
        if any(len(x) > 0 for x in incorrect_dgms)
        else np.empty((0, 2))
    )

    # ----------------------------------------------------------
    # Convert death -> persistence
    # ----------------------------------------------------------

    correct_bp = np.column_stack([
        correct_points[:, 0],
        correct_points[:, 1] - correct_points[:, 0]
    ])

    incorrect_bp = np.column_stack([
        incorrect_points[:, 0],
        incorrect_points[:, 1] - incorrect_points[:, 0]
    ])

    # ----------------------------------------------------------
    # Axis limits
    # ----------------------------------------------------------

    all_points = []

    if len(correct_bp) > 0:
        all_points.append(correct_bp)

    if len(incorrect_bp) > 0:
        all_points.append(incorrect_bp)

    all_points = np.vstack(all_points)

    xmin = all_points[:, 0].min()
    xmax = all_points[:, 0].max()

    ymin = 0
    ymax = all_points[:, 1].max()

    xpad = max((xmax - xmin) * 0.08, 1e-6)
    ypad = max(ymax * 0.08, 1e-6)

    xlim = (
        xmin - xpad,
        xmax + xpad
    )

    ylim = (
        0,
        ymax + ypad
    )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # ----------------------------------------------------------
    # KDE helper
    # ----------------------------------------------------------

    def plot_kde(points, cmap, color):

        if len(points) < 5:
            return

        try:

            values = np.vstack([
                points[:, 0],
                points[:, 1]
            ])

            kde = gaussian_kde(values)

            X, Y = np.meshgrid(
                np.linspace(
                    xlim[0],
                    xlim[1],
                    150
                ),
                np.linspace(
                    ylim[0],
                    ylim[1],
                    150
                )
            )

            positions = np.vstack([
                X.ravel(),
                Y.ravel()
            ])

            Z = kde(
                positions
            ).reshape(X.shape)

            if Z.max() > 0:

                ax.contour(
                    X,
                    Y,
                    Z,
                    levels=contour_levels,
                    colors=color,
                    linewidths=1.2,
                    alpha=0.65
                )

        except (np.linalg.LinAlgError, ValueError):
            pass

    # ----------------------------------------------------------
    # KDE contours
    # ----------------------------------------------------------

    if kde:

        plot_kde(
            correct_bp,
            "Blues",
            "blue"
        )

        plot_kde(
            incorrect_bp,
            "Reds",
            "red"
        )

    # ----------------------------------------------------------
    # Correct
    # ----------------------------------------------------------

    if len(correct_bp) > 0:

        ax.scatter(
            correct_bp[:, 0],
            correct_bp[:, 1],

            c="darkblue",

            s=point_size,

            alpha=alpha,

            marker="^",

            edgecolors="white",

            linewidths=0.5,

            label="Correct",

            zorder=5
        )

    # ----------------------------------------------------------
    # Incorrect
    # ----------------------------------------------------------

    if len(incorrect_bp) > 0:

        ax.scatter(
            incorrect_bp[:, 0],
            incorrect_bp[:, 1],

            c="darkred",

            s=point_size,

            alpha=alpha,

            marker="o",

            edgecolors="white",

            linewidths=0.5,

            label="Incorrect",

            zorder=6
        )

    # ----------------------------------------------------------
    # Formatting
    # ----------------------------------------------------------

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(
        "Birth",
        fontsize=13
    )

    ax.set_ylabel(
        "Persistence (Death − Birth)",
        fontsize=13
    )

    ax.set_title(
        f"Birth–Persistence Overlay — H{dim}",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

def plot_persistence_survival(
    correct_diagrams,
    incorrect_diagrams,
    dim=0,
    figsize=(10, 7),
    normalize=False,
):
    """
    Plot persistence survival curves.

    At each persistence threshold p:

        N(p) = number of features with
               persistence >= p

    normalize=False:
        plots raw number of features

    normalize=True:
        plots fraction of features surviving
    """

    correct_dgms = _collect_diagrams(
        correct_diagrams,
        dim=dim
    )

    incorrect_dgms = _collect_diagrams(
        incorrect_diagrams,
        dim=dim
    )

    # ----------------------------------------------------------
    # Persistence values
    # ----------------------------------------------------------

    correct_persistence = []

    for dgm in correct_dgms:

        if len(dgm) > 0:

            persistence = (
                dgm[:, 1] -
                dgm[:, 0]
            )

            persistence = persistence[
                persistence > 0
            ]

            correct_persistence.extend(
                persistence.tolist()
            )

    incorrect_persistence = []

    for dgm in incorrect_dgms:

        if len(dgm) > 0:

            persistence = (
                dgm[:, 1] -
                dgm[:, 0]
            )

            persistence = persistence[
                persistence > 0
            ]

            incorrect_persistence.extend(
                persistence.tolist()
            )

    correct_persistence = np.asarray(
        correct_persistence
    )

    incorrect_persistence = np.asarray(
        incorrect_persistence
    )

    # ----------------------------------------------------------
    # Common threshold grid
    # ----------------------------------------------------------

    all_persistence = np.concatenate([
        correct_persistence,
        incorrect_persistence
    ])

    thresholds = np.linspace(
        0,
        all_persistence.max(),
        300
    )

    # ----------------------------------------------------------
    # Survival function
    # ----------------------------------------------------------

    correct_survival = np.array([
        np.sum(
            correct_persistence >= p
        )
        for p in thresholds
    ])

    incorrect_survival = np.array([
        np.sum(
            incorrect_persistence >= p
        )
        for p in thresholds
    ])

    # ----------------------------------------------------------
    # Normalize if requested
    # ----------------------------------------------------------

    if normalize:

        if len(correct_persistence) > 0:
            correct_survival = (
                correct_survival /
                len(correct_persistence)
            )

        if len(incorrect_persistence) > 0:
            incorrect_survival = (
                incorrect_survival /
                len(incorrect_persistence)
            )

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.plot(
        thresholds,
        correct_survival,

        color="darkblue",

        linewidth=3,

        label="Correct"
    )

    ax.plot(
        thresholds,
        incorrect_survival,

        color="darkred",

        linewidth=3,

        label="Incorrect"
    )

    # ----------------------------------------------------------
    # Formatting
    # ----------------------------------------------------------

    ax.set_xlabel(
        "Persistence threshold",
        fontsize=13
    )

    if normalize:

        ax.set_ylabel(
            "Fraction of features surviving",
            fontsize=13
        )

    else:

        ax.set_ylabel(
            "Number of features surviving",
            fontsize=13
        )

    ax.set_title(
        f"Persistence Survival Curve — H{dim}",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

    return {
        "thresholds": thresholds,
        "correct": correct_survival,
        "incorrect": incorrect_survival,
    }

def _persistence_surface(
    points,
    xlim,
    ylim,
    gridsize=150,
    sigma=None,
):
    """
    Create a Gaussian-smoothed persistence surface.

    points:
        N x 2 array containing
        [birth, persistence]
    """

    if len(points) == 0:

        X, Y = np.meshgrid(
            np.linspace(
                xlim[0],
                xlim[1],
                gridsize
            ),
            np.linspace(
                ylim[0],
                ylim[1],
                gridsize
            )
        )

        return X, Y, np.zeros_like(X)

    X, Y = np.meshgrid(
        np.linspace(
            xlim[0],
            xlim[1],
            gridsize
        ),
        np.linspace(
            ylim[0],
            ylim[1],
            gridsize
        )
    )

    # ----------------------------------------------------------
    # Automatic bandwidth
    # ----------------------------------------------------------

    if sigma is None:

        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]

        sigma = max(
            xrange,
            yrange
        ) * 0.035

        sigma = max(
            sigma,
            1e-6
        )

    Z = np.zeros_like(X)

    # ----------------------------------------------------------
    # Weight by persistence
    #
    # Long-lived features contribute more.
    # ----------------------------------------------------------

    max_persistence = max(
        points[:, 1].max(),
        1e-12
    )

    weights = (
        points[:, 1] /
        max_persistence
    )

    # ----------------------------------------------------------
    # Gaussian mixture
    # ----------------------------------------------------------

    for (x, y), weight in zip(
        points,
        weights
    ):

        gaussian = np.exp(
            -(
                (X - x) ** 2 +
                (Y - y) ** 2
            ) /
            (2 * sigma ** 2)
        )

        Z += weight * gaussian

    return X, Y, Z


# ==============================================================================
# PERSISTENCE IMAGE COMPARISON
# ==============================================================================

def plot_persistence_image_comparison(
    correct_diagrams,
    incorrect_diagrams,
    dim=0,
    figsize=(15, 5),
    gridsize=150,
    sigma=None,
):
    """
    Plot:

        1. Correct persistence surface
        2. Incorrect persistence surface
        3. Difference: Correct - Incorrect

    Coordinates:

        x = Birth
        y = Persistence
    """

    correct_dgms = _collect_diagrams(
        correct_diagrams,
        dim=dim
    )

    incorrect_dgms = _collect_diagrams(
        incorrect_diagrams,
        dim=dim
    )

    # ----------------------------------------------------------
    # Convert to Birth × Persistence
    # ----------------------------------------------------------

    correct_points = []

    for dgm in correct_dgms:

        if len(dgm) > 0:

            correct_points.append(
                np.column_stack([
                    dgm[:, 0],
                    dgm[:, 1] - dgm[:, 0]
                ])
            )

    incorrect_points = []

    for dgm in incorrect_dgms:

        if len(dgm) > 0:

            incorrect_points.append(
                np.column_stack([
                    dgm[:, 0],
                    dgm[:, 1] - dgm[:, 0]
                ])
            )

    correct_points = (
        np.vstack(correct_points)
        if correct_points
        else np.empty((0, 2))
    )

    incorrect_points = (
        np.vstack(incorrect_points)
        if incorrect_points
        else np.empty((0, 2))
    )

    # ----------------------------------------------------------
    # Common limits
    # ----------------------------------------------------------

    all_points = np.vstack([
        correct_points,
        incorrect_points
    ])

    xmin = all_points[:, 0].min()
    xmax = all_points[:, 0].max()

    ymax = all_points[:, 1].max()

    xpad = max(
        (xmax - xmin) * 0.05,
        1e-6
    )

    ypad = max(
        ymax * 0.05,
        1e-6
    )

    xlim = (
        xmin - xpad,
        xmax + xpad
    )

    ylim = (
        0,
        ymax + ypad
    )

    # ----------------------------------------------------------
    # Surfaces
    # ----------------------------------------------------------

    X, Y, Z_correct = _persistence_surface(
        correct_points,
        xlim,
        ylim,
        gridsize=gridsize,
        sigma=sigma
    )

    _, _, Z_incorrect = _persistence_surface(
        incorrect_points,
        xlim,
        ylim,
        gridsize=gridsize,
        sigma=sigma
    )

    # ----------------------------------------------------------
    # Difference
    # ----------------------------------------------------------

    Z_difference = (
        Z_incorrect -
        Z_correct
    )

    # Symmetric color scale
    difference_limit = np.max(
        np.abs(Z_difference)
    )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=True,
        sharey=True
    )

    # ==========================================================
    # Correct
    # ==========================================================

    im1 = axes[0].contourf(
        X,
        Y,
        Z_correct,
        levels=12,
        cmap="Blues"
    )

    axes[0].contour(
        X,
        Y,
        Z_correct,
        levels=8,
        colors="darkblue",
        linewidths=0.7,
        alpha=0.7
    )

    axes[0].set_title(
        "Correct",
        fontweight="bold"
    )

    # ==========================================================
    # Incorrect
    # ==========================================================

    im2 = axes[1].contourf(
        X,
        Y,
        Z_incorrect,
        levels=12,
        cmap="Reds"
    )

    axes[1].contour(
        X,
        Y,
        Z_incorrect,
        levels=8,
        colors="darkred",
        linewidths=0.7,
        alpha=0.7
    )

    axes[1].set_title(
        "Incorrect",
        fontweight="bold"
    )

    # ==========================================================
    # Difference
    # ==========================================================

    im3 = axes[2].contourf(
        X,
        Y,
        Z_difference,

        levels=15,

        cmap="RdBu_r",

        vmin=-difference_limit,
        vmax=difference_limit
    )

    axes[2].contour(
        X,
        Y,
        Z_difference,

        levels=8,

        colors="black",

        linewidths=0.5,

        alpha=0.5
    )

    axes[2].set_title(
        "Difference\nCorrect − Incorrect",
        fontweight="bold"
    )

    # ==========================================================
    # Labels
    # ==========================================================

    for ax in axes:

        ax.set_xlabel(
            "Birth",
            fontsize=12
        )

        ax.grid(
            True,
            alpha=0.15
        )

    axes[0].set_ylabel(
        "Persistence (Death − Birth)",
        fontsize=12
    )

    # ==========================================================
    # Colorbars
    # ==========================================================

    fig.colorbar(
        im1,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label="Density"
    )

    fig.colorbar(
        im2,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Density"
    )

    fig.colorbar(
        im3,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
        label="Incorrect − Correct"
    )

    fig.suptitle(
        f"Persistence Surface Comparison — H{dim}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

    return {
        "X": X,
        "Y": Y,
        "correct": Z_correct,
        "incorrect": Z_incorrect,
        "difference": Z_difference,
    }

def plot_betti_curve(
    correct_diagrams,
    incorrect_diagrams,
    dim=0,
    figsize=(10, 7),
):
    """
    Plot Betti curves.

    At filtration value t:

        beta(t) = number of persistence intervals
                  containing t
    """

    correct_dgms = _collect_diagrams(
        correct_diagrams,
        dim=dim
    )

    incorrect_dgms = _collect_diagrams(
        incorrect_diagrams,
        dim=dim
    )

    # ----------------------------------------------------------
    # Combine intervals
    # ----------------------------------------------------------

    correct_intervals = (
        np.vstack(correct_dgms)
        if any(len(x) > 0 for x in correct_dgms)
        else np.empty((0, 2))
    )

    incorrect_intervals = (
        np.vstack(incorrect_dgms)
        if any(len(x) > 0 for x in incorrect_dgms)
        else np.empty((0, 2))
    )

    # ----------------------------------------------------------
    # Remove infinite intervals
    # ----------------------------------------------------------

    correct_intervals = correct_intervals[
        np.isfinite(correct_intervals).all(axis=1)
    ]

    incorrect_intervals = incorrect_intervals[
        np.isfinite(incorrect_intervals).all(axis=1)
    ]

    # ----------------------------------------------------------
    # Grid
    # ----------------------------------------------------------

    all_intervals = np.vstack([
        correct_intervals,
        incorrect_intervals
    ])

    xmin = all_intervals[:, 0].min()
    xmax = all_intervals[:, 1].max()

    x_grid = np.linspace(
        xmin,
        xmax,
        500
    )

    # ----------------------------------------------------------
    # Betti curves
    # ----------------------------------------------------------

    correct_betti = np.array([
        np.sum(
            (correct_intervals[:, 0] <= x) &
            (correct_intervals[:, 1] >= x)
        )
        for x in x_grid
    ])

    incorrect_betti = np.array([
        np.sum(
            (incorrect_intervals[:, 0] <= x) &
            (incorrect_intervals[:, 1] >= x)
        )
        for x in x_grid
    ])

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.plot(
        x_grid,
        correct_betti,

        color="darkblue",

        linewidth=3,

        label="Correct"
    )

    ax.plot(
        x_grid,
        incorrect_betti,

        color="darkred",

        linewidth=3,

        label="Incorrect"
    )

    ax.fill_between(
        x_grid,
        correct_betti,
        incorrect_betti,

        where=(
            incorrect_betti >
            correct_betti
        ),

        color="red",
        alpha=0.12
    )

    ax.fill_between(
        x_grid,
        correct_betti,
        incorrect_betti,

        where=(
            correct_betti >
            incorrect_betti
        ),

        color="blue",
        alpha=0.12
    )

    ax.set_xlabel(
        "Filtration value",
        fontsize=13
    )

    ax.set_ylabel(
        f"β{dim}(filtration)",
        fontsize=13
    )

    ax.set_title(
        f"Betti Curve — H{dim}",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

    return {
        "x": x_grid,
        "correct": correct_betti,
        "incorrect": incorrect_betti,
    }

def persistence_diff_acc(datasets, models, type="avg"):
    colors = {"llama": "orange", "qwen": "green", "mistral": "blue"}
    fig, ax = plt.subplots(figsize=(5, 4))
    per_list = []
    acc_list = []
    for dataset in datasets:
        for model in models:
            df = pd.read_csv(f'/Users/kimlopez/TDA_RI/TDA_reason-interpret/{model[2]}/{model[2]}_{dataset[0]}_tda.csv')
            accuracy = df['correctness'].value_counts(normalize=True)[True]
            
            correct_diagrams, incorrect_diagrams = sample_helper(model, dataset, num=100)
            
            # get persistence values for each sample
            correct_persistence = [get_persistence(dgm) for dgm in correct_diagrams]
            incorrect_persistence = [get_persistence(dgm) for dgm in incorrect_diagrams]
            
            # remove irrelevant values
            correct_persistence = [p for p in correct_persistence if len(p) > 0]
            incorrect_persistence = [p for p in incorrect_persistence if len(p) > 0]

            print("persistence: ", correct_persistence)

            # get difference in samples
            # per_diff = [x - y for x, y in zip(correct_persistence, incorrect_persistence)]
            # avg_diff = np.mean(per_diff)
            if type == "avg":
                correct_persistence = np.mean(correct_persistence[0])
                incorrect_persistence = np.mean(incorrect_persistence[0])
            else:
                correct_persistence = [np.sum(p) for p in correct_persistence if len(p) > 0]
                incorrect_persistence = [np.sum(p) for p in incorrect_persistence if len(p) > 0]   

            avg_corr = np.mean(correct_persistence)
            avg_incorr = np.mean(incorrect_persistence)
            avg_diff = abs(avg_corr - avg_incorr)

            per_list.append(avg_diff)
            acc_list.append(accuracy)

            ax.scatter(avg_diff, accuracy,
                    color=colors.get(model[2], "black"),
                    label=dataset, s=35)

            ax.annotate(dataset[0], (avg_diff, accuracy),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8)
    
    # lin reg
    if len(per_list) > 1:
        x = np.asarray(per_list, dtype=float)
        y = np.asarray(acc_list, dtype=float)

        m, b = np.polyfit(x, y, 1)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b

        y_pred = m * x + b
        r2 = r2_score(y, y_pred)

        ax.plot(
            x_line,
            y_line,
            color=colors.get(dataset[0], "black"),
            linestyle="--",
            linewidth=2,
            label=f"{dataset} regression ($R^2={r2:.2f}$)"
        )
    
    ax.set_xlabel("Average Persistence Difference")
    ax.set_ylabel("Accuracy")
    ax.set_title("Persistence Difference vs. Accuracy")

    ax.legend(["llama", "qwen", "mistral"], loc="upper right", fontsize=7)
    ax.grid(False)
    plt.tight_layout()
    plt.show()