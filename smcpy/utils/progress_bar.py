def set_bar(pbar, t, last_ess, ess, acceptance_ratio, resample_status):
    pbar.set_description("Step number: {:2d} | Last ess: {:8.2f} | "
                         "Current ess: {:8.2f} | Samples accepted: "
                         "{:.1%} | {} | "
                         .format(t + 1, last_ess, ess, acceptance_ratio,
                                 resample_status))
    return pbar
