import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.interpolate import interp1d
from smcpy.utils.storage import HDF5Storage
from run_example import *
from generate_data import gen_data_from_mvn, plot_noisy_data, X

N_SAMPLES = 5000
PLOT_DIR = Path(__file__).parent

def get_mean_and_cov(params):
    mvn = MVNHierarchModel(params)
    means = mvn._inputs
    covs = mvn.build_cov_array()
    return means, covs


def get_smc_predictive_samples(step):
    rng = np.random.default_rng()#seed=34)
    means, covs = get_mean_and_cov(step.params)
    samples = np.zeros((means.shape))
    for i, (m, c) in enumerate(zip(means, covs)):
        samples[i, :] = rng.multivariate_normal(m, c)
    df = pd.DataFrame(data=samples, columns=['a', 'b'])
    df['src'] = 'SMC'
    return df


def get_ground_truth_samples(nsamples):
    rng = np.random.default_rng(seed=34)
    ground_truth_samples = rng.multivariate_normal(
        TRUE_PARAMS[0],
        TRUE_COV,
        nsamples
    )
    df = pd.DataFrame(data=ground_truth_samples, columns=['a', 'b'])
    df['src'] = 'Ground Truth'
    return df


def plot_hyperparms(results):
    df = pd.DataFrame(results[-1].param_dict)
    sns.pairplot(df, diag_kind='kde')
    plt.savefig(PLOT_DIR / 'hyperparam_pairplot.png')
 

def plot_predictive(results):
    _, r_effs = gen_data_from_mvn(plot=False)
    df = get_smc_predictive_samples(results[-1])
    gt_df = get_ground_truth_samples(nsamples=df.shape[0])
    sns.pairplot(data=pd.concat((df, gt_df)), kind='kde', hue='src')
    ax = plt.gcf().axes[2]
    ax.plot(r_effs[:, 0], r_effs[:, 1], 'k+')
    plt.savefig(PLOT_DIR / 'predictive_pairplot.png')


def plot_95pct_pred_interval(results):
    df = get_smc_predictive_samples(results[-1])
    gen_data_from_mvn(show=False)
    out = eval_model(df.to_numpy())
    out.sort(axis=0)
    interp = interp1d(np.linspace(0, 1, out.shape[0]), out, axis=0)
    intervals = interp([0.025, 0.5, 0.975])
    plt.fill_between(X, intervals[0], intervals[2], zorder=10, alpha=0.7)
    plt.savefig(PLOT_DIR / 'pred_interval.png')


def plot_hyperparam_animation(results):
    num_steps = len(results)

    df = pd.DataFrame(results[0].param_dict)
    g = sns.PairGrid(data=df)
    xlims = []
    ylims = []

    def prep_axes(g, xlims, ylims):
        for j, ax in enumerate(g.axes.flatten()):
            ax.clear()
            if xlims != []:
                ax.set_xlim(xlims[j])
                ax.set_ylim(ylims[j])
        try:
            for j, ax in enumerate(g.diag_axes.flatten()):
                ax.clear()
                ax.set_xlabel(None)
                ax.set_ylabel(None)
                if xlims != []:
                    ax.set_xlim(xlims[j])
        except AttributeError:
            pass

    def animate(i):
        g.data = pd.DataFrame(results[i].param_dict)
        prep_axes(g, xlims, ylims)
        g.map_lower(sns.scatterplot)
        g.map_upper(sns.kdeplot)
        g.map_diag(sns.kdeplot)
        if xlims == []:
            _ = [xlims.append(x.get_xlim()) for x in g.axes.flatten()]
            _ = [ylims.append(x.get_ylim()) for x in g.axes.flatten()]

    ani = animation.FuncAnimation(g.fig, animate, frames=np.arange(num_steps),
                                  repeat=False, interval=100)
    ani.save(PLOT_DIR / 'hyperparam_pairplot.gif', writer='pillow')


if __name__ == '__main__':

    results = HDF5Storage(HDF5_FILE)

    plot_hyperparms(results)
    plot_predictive(results)
    plot_95pct_pred_interval(results)
    plot_hyperparam_animation(results)