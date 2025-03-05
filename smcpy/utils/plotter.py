"""
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.

Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRessED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNess FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLess THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
"""

import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde


def _mpi_decorator(func):
    def wrapper(self, *args, **kwargs):
        """
        Detects whether multiple processors are available and sets
        self.number_CPUs and self.cpu_rank accordingly. Only calls decorated
        function using rank 0.
        """
        try:
            imp.find_module("mpi4py")

            from mpi4py import MPI

            comm = MPI.COMM_WORLD.Clone()

            size = comm.size
            rank = comm.rank
            comm = comm

        except ImportError:
            size = 1
            rank = 0
            comm = SingleRankComm()

        if rank == 0:
            func(self, *args, **kwargs)

    return wrapper


def plot_mcmc_chain(
    chain,
    param_labels,
    burnin=0,
    save=False,
    show=True,
    include_kde=False,
    filename="chain.png",
    report_style=False,
):
    n_columns = 2
    gridspec = {"width_ratios": [1.0, 0.0], "wspace": 0.0}
    if include_kde:
        gridspec = {"width_ratios": [0.85, 0.15], "wspace": 0.0}

    fig, ax = plt.subplots(
        len(param_labels), n_columns, sharey="row", gridspec_kw=gridspec
    )

    chain = chain[:, :, burnin:]
    for i, name in enumerate(param_labels):
        for parallel_chain in chain:
            ax[i, 0].plot(parallel_chain[i], "-", linewidth=0.5)

            if include_kde:
                ylims = ax[i, 0].get_ylim()
                x = np.linspace(ylims[0], ylims[1], 1000)
                kde = gaussian_kde(parallel_chain[i])
                ax[i, 1].plot(kde.pdf(x), x, "-")
                ax[i, 1].fill_betweenx(x, kde.pdf(x), np.zeros(x.shape), alpha=0.3)

        ax[i, 1].axis("off")
        if include_kde:
            ax[i, 1].set_xlim(0, None)

        ax[i, 0].set_ylabel(name)
        ax[i, 0].set_xlim(0, chain.shape[2])
        ax[i, 0].get_xaxis().set_visible(False)

    ax[len(param_labels) - 1, 0].get_xaxis().set_visible(True)
    ax[len(param_labels) - 1, 0].set_xlabel("sample #")
    ax[len(param_labels) - 1, 1].set_xlabel("probability density")

    if save:
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
    if show:
        plt.show()

    if report_style:
        fig.set_figheight(fig.get_figheight() * ax.shape[0])
        rprt_filename = os.path.splitext(filename)[0] + "_report.pdf"
        with PdfPages(rprt_filename) as pdf:
            scale = fig.dpi_scale_trans.inverted()
            extent1 = ax[0, 0].get_window_extent().transformed(scale)
            extent2 = ax[1, 0].get_window_extent().transformed(scale)
            ax_height = extent1.y1 - extent2.y1

            for i in range(ax.shape[0]):
                ax[i, 0].get_xaxis().set_visible(True)
                ax[i, 0].set_xlabel("sample #")
                ax[i, 1].set_xlabel("probability density")
                extent = ax[i, 0].get_window_extent().transformed(scale)
                extent.x0 = 0
                extent.x1 = fig.get_figwidth()
                extent.y0 = extent.y1 - ax_height
                pdf.savefig(plt.gcf(), bbox_inches=extent)

    return fig


def plot_pairwise(
    samples,
    weights=None,
    param_names=None,
    save=False,
    show=True,
    xlim=None,
    ylim=None,
    filename="pairwise.png",
):  # pragma no cover
    # TODO true params on plot #true_params=None,
    """
    Plots pairwise distributions of all parameter combos. Color codes each
    by weight if provided.
    """
    if param_names is None:
        param_names = [f"p{i}" for i in range(samples.shape[1])]

    if weights is None:
        weights = np.ones((samples.shape[0], 1))

    columns = param_names + ["weights_"]
    samples = np.hstack((samples, weights))

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([])

    df = pd.DataFrame(samples, columns=columns)

    ax = sns.pairplot(
        df,
        diag_kind="kde",
        corner=True,
        hue="weights_",
        palette="viridis",
        diag_kws={"weights": weights.flatten(), "hue": None},
    )
    ax.set(xlim=xlim)
    ax.set(ylim=ylim)

    ax.legend.remove()
    cbar = plt.gcf().colorbar(sm)
    cbar.ax.set_ylabel("Particle Weight")

    # compute means TODO plot means
    if weights is None:
        means = np.mean(samples, axis=0)
    else:
        means = np.sum(samples * weights, axis=0)

    if save:
        plt.savefig(filename)
    if show:
        plt.show()

    return ax


def plot_geweke(burnin, z, param_labels=None):
    n_params = z[0].shape[0]
    if param_labels is None:
        param_labels = [f"Param{i}" for i in range(n_params)]

    xlim = (0, int(burnin[-1] * 1.1))
    y = np.ones(2)

    fig, ax = plt.subplots(n_params)
    for i in range(n_params):
        ax[i].fill_between([0, xlim[1]], y * 2, y * -2, alpha=0.5, color="0.4")
        ax[i].fill_between([0, xlim[1]], y * 1, y * -1, alpha=0.5, color="0.4")
        ax[i].axhline(2, linestyle="--", color="k")
        ax[i].axhline(-2, linestyle="--", color="k")
        ax[i].plot(burnin, z[:, i], "o")
        ax[i].set_xlim(xlim)
        ax[i].set_ylabel(param_labels[i])

    ax[0].set_title("Geweke Scores")
    ax[-1].set_xlabel("Burnin")
    plt.show()
