import matplotlib.pyplot as plt
import numpy as np
import os
import json
'''
Plotting functions
'''

def plot_target_boxplots(true_values, run_label, **series):
    """Plot box plots for each parameter and series over the phi sequence.

    Args:
        true_values: array-like of true parameter values in param order
        **series: label=(targets_list, phi_sequence) for each SMC run
    """
    first_targets, _ = next(iter(series.values()))
    param_names = first_targets[0].param_names
    n_params = len(param_names)
    n_series = len(series)

    fig, axes = plt.subplots(
        n_params,
        n_series,
        sharex="col",
        sharey="row",
        figsize=(4 * n_series, 3 * n_params),
        squeeze=False
    )
    axes = np.atleast_2d(axes)

    for col, (label, (targets, phi_sequence)) in enumerate(series.items()):
        positions = np.arange(len(phi_sequence))
        box_width = 0.6
        for row, (name, true_val) in enumerate(zip(param_names, true_values)):
            ax = axes[row, col]
            ax.boxplot(
                [target.params[:, row] for target in targets],
                positions=positions,
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
            )
            ax.axhline(
                true_val,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label="true value",
                alpha=0.7,
            )
            ax.tick_params(axis='y', labelrotation=45)
            ax.grid(True)
            if col == 0:
                ax.set_ylabel(f"${name}$")
        axes[0, col].set_title(label + f" ({len(targets)} steps)")

    for col in range(n_series):
        axes[-1, col].set_xlabel("step")

    axes[0, -1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{run_label}_smc_steps.png", bbox_inches='tight')
    plt.show()

def plot_2d_joint_posterior(true_values, run_label, **series):
    """
    Plots a 2D scatter of the joint posterior particles for each SMC run.

    Args:
        true_values: Array-like of true parameter values (needs at least 2 values).
        **series: label=(targets_list, phi_sequence) for each SMC run.
    """
    n_series = len(series)
    
    # Extract true values from the array (fixes the undefined variables bug)
    theta_0, theta_1 = true_values[0], true_values[1]

    # Create subplots
    fig, axes = plt.subplots(
        1,
        n_series,
        sharex="col",
        sharey="row",
        figsize=(6 * n_series, 6), # slightly adjusted size for better proportions
        # squeeze=False
    )
    
    # Ensure axes is a 1D array so we can index it cleanly, even if n_series == 1
    axes = np.atleast_1d(axes)

    # Unpack the series tuple directly in the loop
    for col, (label, targets) in enumerate(series.items()):
        ax = axes[col]
        
        # Plot posterior particles
        ax.scatter(
            targets[-1].params[:, 0],
            targets[-1].params[:, 1],
            alpha=0.3,           
            s=20,                
            color='mediumblue',  
            edgecolors='none',   
            label='Posterior Particles'
        )

        # Plot True values lines using the specific Axis object (ax)
        ax.axvline(theta_0, color='r', linestyle='--', alpha=0.8, linewidth=2)
        ax.axhline(theta_1, color='r', linestyle='--', alpha=0.8, linewidth=2)

        # Add a marker right at the exact true value intersection
        ax.plot(theta_0, theta_1, marker='*', color='red', markersize=15, 
                linestyle='None', label='True Theta')

        # Labels and Titles
        ax.set_xlabel(r'$\theta_0$', fontsize=14)
        ax.set_title(label, fontsize=16, pad=15)
        
        # Only add the y-label to the first plot since the y-axis is shared
        if col == 0:
            ax.set_ylabel(r'$\theta_1$', fontsize=14)

        # Add a subtle background grid for readability
        ax.grid(True, alpha=0.5)

        # Make tick marks slightly larger
        ax.tick_params(axis='both', which='major', labelsize=12, labelrotation = 45)

    # Add the legend to the last axis
    axes[-1].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(f"{run_label}_joint_posterior.png", bbox_inches='tight')
    plt.show()

def plot_param_hists(true_values, run_label, **series):
    """
    Plots a histogram of the marginal posterior for each parameter (columns) 
    across each SMC run (rows).

    Args:
        true_values: Array-like of true parameter values.
        **series: label=(targets_list, phi_sequence) for each SMC run.
    """
    n_runs = len(series)
    n_params = len(true_values)
    
    # Create subplots: n_runs (rows) x n_params (columns)
    # squeeze=False ensures `axes` is strictly a 2D array: shape (n_runs, n_params)
    fig, axes = plt.subplots(
        n_runs,
        n_params,
        sharex="col",  # X-axis shared across the same parameter (columns)
        figsize=(8 * n_params, 6 * n_runs), 
        squeeze=False 
    )

    for row, (label, series_data) in enumerate(series.items()):
        # Handle unpacking just in case `series_data` is a tuple (targets, phi_sequence)
        if isinstance(series_data, tuple) and len(series_data) == 2:
            targets = series_data[0]
        else:
            targets = series_data
            
        # Extract the final particles for this specific SMC run
        final_particles = targets[-1].params
        
        for col in range(n_params):
            ax = axes[row, col]
            
            # Plot the histogram for this specific parameter
            ax.hist(
                final_particles[:, col],
                bins=30,
                # density=True,      # Normalizes the histogram to represent a probability density
                alpha=0.7,           
                label='Posterior Density' if row == 0 and col == 0 else None,
                color = 'mediumblue'
            )

            # Plot True value line
            true_val = true_values[col]
            ax.axvline(true_val, color='r', linestyle='--', alpha=0.8, linewidth=2, 
                       label='True Value' if row == 0 and col == 0 else None)

            # --- Formatting and Labels ---
            
            # Put the parameter symbol as the title on the top row
            if row == 0:
                ax.set_title(r'$\theta_{%d}$' % col, fontsize=16, pad=10)
            
            # Put the parameter symbol on the x-axis of the bottom row
            if row == n_runs - 1:
                ax.set_xlabel(r'$\theta_{%d}$' % col, fontsize=14)
            
            # Put the SMC run label on the y-axis of the first column
            if col == 0:
                ax.set_ylabel(f'{label}\nFrequency', fontsize=14)

            ax.tick_params(axis='x', labelrotation=45)

            # Clean up the background grid and ticks
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=11)
            # Turn off scientific notation and the base offset
            ax.ticklabel_format(useOffset=False, style='plain', axis='x')

    axes[0,0].legend(loc="upper right")

    
    plt.tight_layout()
    # Adjust top margin so the figure legend doesn't overlap titles
    plt.subplots_adjust(top=0.9) 
    plt.savefig(f"{run_label}_histogram_comp.png", bbox_inches='tight')
    plt.show()

def plot_ill_posed_res(true_values, run_label, perturbed_lofi_particles, bias_adjustments_arr, stdev_adjustments_arr, **series):
    n_params = len(true_values)
    
    # Create subplots: n_runs (rows) x n_params (columns)
    # squeeze=False ensures `axes` is strictly a 2D array: shape (n_runs, n_params)
    fig, axes = plt.subplots(
        2,
        n_params*2,
        figsize=(10 * n_params, 6 * 2), 
        squeeze=False 
    )

    targets, phi_sequence = next(iter(series.values()))
    param_names = targets[0].param_names
    positions = np.arange(len(phi_sequence))
    box_width = 0.6

    for param_ind in range(n_params):
        
        # plot smc steps 
        ax = axes[0, param_ind * 2]
        ax.boxplot(
            [target.params[:, param_ind] for target in targets],
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
        )
        ax.axhline(
            true_values[param_ind],
            color="r",
            linestyle="--",
            linewidth=1.5,
            label="true value",
            alpha=0.7,
        )
        ax.tick_params(axis='y', labelrotation=45)
        ax.grid(True)
        ax.set_xlabel("SMC Cycles", fontsize=14)
        ax.set_ylabel(f"${param_names[param_ind]}$", fontsize=14)
        ax.set_title(f" {param_names[param_ind]} | ({len(targets)} steps)", fontsize=14)

        # plot lofi posterior
        ax = axes[1, param_ind * 2]
            
        # Plot the histogram for this specific parameter
        ax.hist(
            perturbed_lofi_particles[:, param_ind],
            bins=30,
            # density=True,      # Normalizes the histogram to represent a probability density
            alpha=0.7,           
            color = 'mediumblue'
        )

        # Plot True value line
        true_val = true_values[param_ind]
        ax.axvline(true_val, color='r', linestyle='--', alpha=0.8, linewidth=2)

        # --- Formatting and Labels ---
        ax.set_title('LoFi Posterior', fontsize=16, pad=10)
        ax.set_xlabel(param_names[param_ind], fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.tick_params(axis='x', labelrotation=45)

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)

        # plot HiFi posterior
        final_particles = targets[-1].params
        ax = axes[1, (param_ind * 2) + 1]
        # Plot the histogram for this specific parameter
        ax.hist(
            final_particles[:, param_ind],
            bins=30,
            # density=True,      # Normalizes the histogram to represent a probability density
            alpha=0.7,           
            color = 'mediumblue'
        )

        # Plot True value line
        true_val = true_values[param_ind]
        ax.axvline(true_val, color='r', linestyle='--', alpha=0.8, linewidth=2)

        # --- Formatting and Labels ---
        ax.set_title('HiFi Posterior', fontsize=16, pad=10)
        ax.set_xlabel(param_names[param_ind], fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.tick_params(axis='x', labelrotation=45)

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)
    
    # plot LoFi joint posterior
    ax = axes[0, 1]
    ax.scatter(
        perturbed_lofi_particles[:, 0],
        perturbed_lofi_particles[:, 1],
        alpha=0.3,           
        s=20,                
        color='mediumblue',  
        edgecolors='none',   
        label='Posterior Particles'
    )

    # Plot True values lines using the specific Axis object (ax)
    ax.axvline(true_values[0], color='r', linestyle='--', alpha=0.8, linewidth=2)
    ax.axhline(true_values[1], color='r', linestyle='--', alpha=0.8, linewidth=2)

    # Add a marker right at the exact true value intersection
    ax.plot(true_values[0], true_values[1], marker='*', color='red', markersize=15, 
            linestyle='None', label='True Theta')

    # Labels and Titles
    ax.set_title("Low-Fidelity", fontsize=14)
    ax.set_xlabel(r'$\theta_0$', fontsize=14)
    ax.set_ylabel(r'$\theta_1$', fontsize=14)
    ax.grid(True, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12, labelrotation = 45)

    # plot HiFi joint posterior
    ax = axes[0, 3]
    ax.scatter(
        targets[-1].params[:, 0],
        targets[-1].params[:, 1],
        alpha=0.3,           
        s=20,                
        color='mediumblue',  
        edgecolors='none',   
        label='Posterior Particles'
    )

    # Plot True values lines using the specific Axis object (ax)
    ax.axvline(true_values[0], color='r', linestyle='--', alpha=0.8, linewidth=2)
    ax.axhline(true_values[1], color='r', linestyle='--', alpha=0.8, linewidth=2)

    # Add a marker right at the exact true value intersection
    ax.plot(true_values[0], true_values[1], marker='*', color='red', markersize=15, 
            linestyle='None', label='True Theta')

    # Labels and Titles
    ax.set_title("High-Fidelity", fontsize=14)
    ax.set_xlabel(r'$\theta_0$', fontsize=14)
    ax.set_ylabel(r'$\theta_1$', fontsize=14)
    ax.grid(True, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12, labelrotation = 45)
    
    title = f"Bias Adjustments: {bias_adjustments_arr}  |  Std Dev Adjustments: {stdev_adjustments_arr}"
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    # Adjust top margin so the figure legend doesn't overlap titles
    plt.subplots_adjust(top=0.9) 
    plt.savefig(f"{run_label}_results.png", bbox_inches='tight')
    plt.show()
    return None

def plot_log_likelihood(run_label, **series):
    """
    Plots the Marginal Log-Likelihood (MLL) progression across SMC steps.
    
    Args:
        run_label (str): A prefix used for saving the output plot filename.
        **series: Keyword arguments where the key is the label (e.g., run name) 
                  and the value is an iterable/tuple where the second element 
                  is the MLL array over steps.
    """
    n_series = len(series)
    
    # Create the subplots. squeeze=False ensures axes is ALWAYS a 2D array 
    # of shape (1, n_series), even if n_series is 1.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_series,
        sharex="col",
        sharey="row",
        figsize=(5 * n_series, 4), # Slightly taller and wider for a cleaner look
        squeeze=False 
    )

    for col, (label, mll_arr) in enumerate(series.items()):
        # Extract the MLL array (assuming series_data is a tuple of (targets, mll_arr))
        positions = np.arange(len(mll_arr))

        # Access the specific subplot (Row 0, Column 'col')
        ax = axes[0, col]
        
        # Plot the line and scatter points
        # Added zorder to ensure points sit on top of the line
        line = ax.plot(positions, mll_arr, linewidth=2, alpha=0.8, label="MLL Trajectory")
        ax.scatter(positions, mll_arr, s=30, color=line[0].get_color(), zorder=3)

        # Styling and Labels
        ax.set_title(f"{label}", fontsize=12, fontweight='bold')
        ax.set_xlabel("SMC Step", fontsize=11)
        
        # Only add the Y-label to the far-left plot to avoid clutter (since sharey=True)
        if col == 0:
            ax.set_ylabel("Marginal Log-Likelihood", fontsize=11)
            
        # Add a subtle background grid for easier reading
        ax.grid(True, linestyle='--', alpha=0.6)

    # Add a legend to the final subplot
    axes[0, -1].legend(loc="best", fontsize=10)

    # Clean layout and save
    plt.tight_layout()
    file_name = f"{run_label}_mll_per_smc_step.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300) # dpi=300 makes the image high-res
    
    plt.show()


def save_run_hyperparameters(log_filepath, run_label, **hyperparameters):

    # 2. Load existing data if the file already exists
    if os.path.exists(log_filepath):
        with open(log_filepath, "r") as log_file:
            try:
                all_runs = json.load(log_file)
            except json.JSONDecodeError:
                all_runs = {} # If the file is empty or corrupted, start fresh
    else:
        all_runs = {}

    # 3. Add or Overwrite the data for this specific run_label
    # (If run_label already exists in the dictionary, this overwrites it. If not, it adds it.)
    all_runs[run_label] = hyperparameters

    # 4. Save the updated dictionary back to the JSON file
    with open(log_filepath, "w") as log_file:
        # indent=4 makes the JSON file nicely formatted and easy for humans to read!
        json.dump(all_runs, log_file, indent=4)

    print(f"Run details successfully logged to {log_filepath}")

    return None