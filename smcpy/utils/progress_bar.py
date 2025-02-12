from tqdm import tqdm


def set_bar(pbar, t, mutation_ratio, updater):
    if isinstance(pbar, tqdm):
        desc = "t: {:2d} | ess: {:8.2f} | mut. ratio: {:.1%} | resample: {} |"
        pbar.set_description(
            desc.format(t, updater.ess, mutation_ratio, updater.resampled)
        )
    return pbar
