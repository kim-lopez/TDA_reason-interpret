import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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

# def plot_barcode_overlay(
#     correct_diagrams,
#     incorrect_diagrams,
#     dim=1,
#     figsize=(12, 8),
#     alpha=0.65,
#     linewidth=4,
#     max_intervals=None,
# ):
#     """
#     Overlay persistence barcodes for correct and incorrect samples.

#     Correct intervals are blue.
#     Incorrect intervals are red.

#     Args:
#         correct_diagrams:
#             Persistence diagrams returned by:
#                 ripser(..., maxdim=1)["dgms"]

#         incorrect_diagrams:
#             Persistence diagrams returned by:
#                 ripser(..., maxdim=1)["dgms"]

#         dim:
#             Homology dimension to plot. Use dim=1 for H1.

#         figsize:
#             Figure size.

#         alpha:
#             Transparency of barcode intervals.

#         linewidth:
#             Width of barcode intervals.

#         max_intervals:
#             Maximum number of intervals to plot from each condition.
#             Longest-persistence intervals are retained.
#     """

#     # ================================================================
#     # Get persistence diagrams for requested homology dimension
#     # ================================================================

#     correct_dgm = np.asarray(correct_diagrams[dim])
#     incorrect_dgm = np.asarray(incorrect_diagrams[dim])

#     # ================================================================
#     # Remove infinite intervals
#     # ================================================================

#     correct_finite = correct_dgm[
#         np.isfinite(correct_dgm).all(axis=1)
#     ]

#     incorrect_finite = incorrect_dgm[
#         np.isfinite(incorrect_dgm).all(axis=1)
#     ]

#     # ================================================================
#     # Sort by persistence
#     # Longest intervals first
#     # ================================================================

#     if len(correct_finite) > 0:
#         correct_persistence = (
#             correct_finite[:, 1] - correct_finite[:, 0]
#         )

#         correct_finite = correct_finite[
#             np.argsort(-correct_persistence)
#         ]

#     if len(incorrect_finite) > 0:
#         incorrect_persistence = (
#             incorrect_finite[:, 1] - incorrect_finite[:, 0]
#         )

#         incorrect_finite = incorrect_finite[
#             np.argsort(-incorrect_persistence)
#         ]

#     # ================================================================
#     # Limit number of intervals if requested
#     # ================================================================

#     if max_intervals is not None:
#         correct_finite = correct_finite[:max_intervals]
#         incorrect_finite = incorrect_finite[:max_intervals]

#     # ================================================================
#     # Determine common x-axis
#     # ================================================================

#     all_intervals = []

#     if len(correct_finite) > 0:
#         all_intervals.append(correct_finite)

#     if len(incorrect_finite) > 0:
#         all_intervals.append(incorrect_finite)

#     if len(all_intervals) > 0:

#         all_intervals = np.vstack(all_intervals)

#         xmin = all_intervals[:, 0].min()
#         xmax = all_intervals[:, 1].max()

#         padding = max(
#             (xmax - xmin) * 0.05,
#             0.01
#         )

#         xmin -= padding
#         xmax += padding

#     else:
#         xmin = 0
#         xmax = 1

#     # ================================================================
#     # Create plot
#     # ================================================================

#     fig, ax = plt.subplots(figsize=figsize)

#     # ------------------------------------------------
#     # Put correct and incorrect on the SAME y-axis
#     # ------------------------------------------------

#     # Use separate y ranges so that intervals from the
#     # two conditions overlap visually.

#     n_correct = len(correct_finite)
#     n_incorrect = len(incorrect_finite)

#     # Correct intervals occupy one vertical region
#     correct_y = np.arange(n_correct)

#     # Incorrect intervals occupy the same vertical region
#     # so that intervals can visually overlap.
#     incorrect_y = np.linspace(
#         0,
#         max(n_correct - 1, n_incorrect - 1),
#         n_incorrect
#     ) if n_incorrect > 0 else []

#     # ================================================================
#     # Plot CORRECT — BLUE
#     # ================================================================

#     for i, (birth, death) in enumerate(correct_finite):

#         ax.plot(
#             [birth, death],
#             [correct_y[i], correct_y[i]],
#             color="blue",
#             linewidth=linewidth,
#             alpha=alpha,
#             solid_capstyle="round",
#             zorder=2
#         )

#         # Birth marker
#         ax.scatter(
#             birth,
#             correct_y[i],
#             color="darkblue",
#             s=30,
#             zorder=3
#         )

#         # Death marker
#         ax.scatter(
#             death,
#             correct_y[i],
#             color="darkblue",
#             s=30,
#             zorder=3
#         )

#     # ================================================================
#     # Plot INCORRECT — RED
#     # ================================================================

#     for i, (birth, death) in enumerate(incorrect_finite):

#         ax.plot(
#             [birth, death],
#             [incorrect_y[i], incorrect_y[i]],
#             color="red",
#             linewidth=linewidth,
#             alpha=alpha,
#             solid_capstyle="round",
#             zorder=4
#         )

#         # Birth marker
#         ax.scatter(
#             birth,
#             incorrect_y[i],
#             color="darkred",
#             s=30,
#             zorder=5
#         )

#         # Death marker
#         ax.scatter(
#             death,
#             incorrect_y[i],
#             color="darkred",
#             s=30,
#             zorder=5
#         )

#     # ================================================================
#     # Legend
#     # ================================================================

#     ax.plot(
#         [],
#         [],
#         color="blue",
#         linewidth=linewidth,
#         label="Correct"
#     )

#     ax.plot(
#         [],
#         [],
#         color="red",
#         linewidth=linewidth,
#         label="Incorrect"
#     )

#     # ================================================================
#     # Formatting
#     # ================================================================

#     ax.set_xlim(xmin, xmax)

#     max_y = max(
#         n_correct,
#         n_incorrect,
#         1
#     )

#     ax.set_ylim(
#         -1,
#         max_y
#     )

#     ax.set_xlabel(
#         "Filtration value",
#         fontsize=13
#     )

#     ax.set_ylabel(
#         "Persistence interval",
#         fontsize=13
#     )

#     ax.set_title(
#         f"Correct vs Incorrect Persistence Barcode — H{dim}",
#         fontsize=15,
#         fontweight="bold"
#     )

#     ax.legend(
#         loc="upper right",
#         fontsize=11
#     )

#     ax.grid(
#         axis="x",
#         alpha=0.25
#     )

#     plt.tight_layout()
#     plt.show()

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
    Plot mean persistence landscapes for correct vs incorrect samples.

    Each persistence diagram is converted into a persistence landscape.
    The landscapes are then averaged pointwise across all samples.

    Correct  = blue
    Incorrect = red

    Args:
        correct_diagrams:
            List of persistence diagrams returned by:
                ripser(..., maxdim=1)["dgms"]

        incorrect_diagrams:
            List of persistence diagrams returned by:
                ripser(..., maxdim=1)["dgms"]

        dim:
            Homology dimension. Use dim=1 for H1.

        num_layers:
            Number of persistence landscape layers to compute.

        num_points:
            Number of points in the filtration grid.

        figsize:
            Matplotlib figure size.

        show_std:
            Whether to show +/- 1 standard deviation bands.

    Returns:
        results:
            Dictionary containing the x-axis and mean/std landscapes.
    """

    # ================================================================
    # Helper: convert one persistence diagram into a landscape
    # ================================================================

    def compute_landscape(dgm, x_grid, num_layers):
        """
        Compute persistence landscape for one persistence diagram.

        For each interval [birth, death], the tent function is:

                  death-birth
                       /\
                      /  \
                     /    \
                    /      \
                   birth   death

        lambda_k(x) is the k-th largest tent value at x.
        """

        dgm = np.asarray(dgm)

        # Remove infinite intervals
        dgm = dgm[
            np.isfinite(dgm).all(axis=1)
        ]

        if len(dgm) == 0:
            return np.zeros(
                (num_layers, len(x_grid))
            )

        # Store tent functions
        tents = []

        for birth, death in dgm:

            if death <= birth:
                continue

            midpoint = (birth + death) / 2
            height = (death - birth) / 2

            # Tent function:
            # max(0, min(x-birth, death-x))
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

        tents = np.asarray(tents)

        # Sort tent functions at every x
        # Largest = lambda_1
        # Second largest = lambda_2
        # etc.
        tents = np.sort(
            tents,
            axis=0
        )[::-1]

        # Keep requested number of layers
        if tents.shape[0] >= num_layers:
            landscape = tents[:num_layers]
        else:
            landscape = np.zeros(
                (num_layers, len(x_grid))
            )

            landscape[:tents.shape[0]] = tents

        return landscape

    # ================================================================
    # Get diagrams for requested dimension
    # ================================================================

    correct_dgms = [
        np.asarray(dgm[dim])
        for dgm in correct_diagrams
    ]

    incorrect_dgms = [
        np.asarray(dgm[dim])
        for dgm in incorrect_diagrams
    ]

    # ================================================================
    # Determine common filtration range
    # ================================================================

    all_points = []

    for dgm in correct_dgms + incorrect_dgms:

        finite = dgm[
            np.isfinite(dgm).all(axis=1)
        ]

        if len(finite) > 0:
            all_points.append(finite)

    if len(all_points) == 0:
        raise ValueError(
            f"No finite persistence points found for H{dim}."
        )

    all_points = np.vstack(all_points)

    xmin = all_points[:, 0].min()
    xmax = all_points[:, 1].max()

    # Add a little padding
    padding = max(
        (xmax - xmin) * 0.05,
        1e-6
    )

    xmin -= padding
    xmax += padding

    # Common filtration grid
    x_grid = np.linspace(
        xmin,
        xmax,
        num_points
    )

    # ================================================================
    # Compute landscapes for every sample
    # ================================================================

    correct_landscapes = np.asarray([
        compute_landscape(
            dgm,
            x_grid,
            num_layers
        )
        for dgm in correct_dgms
    ])

    incorrect_landscapes = np.asarray([
        compute_landscape(
            dgm,
            x_grid,
            num_layers
        )
        for dgm in incorrect_dgms
    ])

    # Shape:
    #
    #   samples × layers × x
    #
    # Average across samples
    # ================================================================

    correct_mean = correct_landscapes.mean(axis=0)
    incorrect_mean = incorrect_landscapes.mean(axis=0)

    correct_std = correct_landscapes.std(axis=0)
    incorrect_std = incorrect_landscapes.std(axis=0)

    # ================================================================
    # Plot
    # ================================================================

    fig, ax = plt.subplots(figsize=figsize)

    colors_correct = [
        "darkblue",
        "royalblue",
        "cornflowerblue",
        "steelblue",
        "lightsteelblue",
    ]

    colors_incorrect = [
        "darkred",
        "red",
        "firebrick",
        "tomato",
        "lightcoral",
    ]

    # ---------------------------------------------------------------
    # Plot each landscape layer
    # ---------------------------------------------------------------

    for k in range(num_layers):

        # Correct
        ax.plot(
            x_grid,
            correct_mean[k],
            color=colors_correct[
                min(k, len(colors_correct) - 1)
            ],
            linewidth=2,
            label=(
                "Correct"
                if k == 0
                else None
            ),
            alpha=0.9
        )

        # Incorrect
        ax.plot(
            x_grid,
            incorrect_mean[k],
            color=colors_incorrect[
                min(k, len(colors_incorrect) - 1)
            ],
            linewidth=2,
            label=(
                "Incorrect"
                if k == 0
                else None
            ),
            alpha=0.9
        )

    # ================================================================
    # Standard deviation bands
    # ================================================================

    if show_std:

        # ------------------------------------------------------------
        # Correct
        # ------------------------------------------------------------

        ax.fill_between(
            x_grid,
            np.maximum(
                0,
                correct_mean[0] - correct_std[0]
            ),
            correct_mean[0] + correct_std[0],
            color="blue",
            alpha=0.12,
            label="Correct ±1 SD"
        )

        # ------------------------------------------------------------
        # Incorrect
        # ------------------------------------------------------------

        ax.fill_between(
            x_grid,
            np.maximum(
                0,
                incorrect_mean[0] - incorrect_std[0]
            ),
            incorrect_mean[0] + incorrect_std[0],
            color="red",
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
        "Persistence landscape λₖ",
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
    # Return results for further analysis
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
