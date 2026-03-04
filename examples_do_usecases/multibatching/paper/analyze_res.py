import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from discrete_optimization.generic_tools.study import Hdf5Database


# 1. Helper function to parse solver configurations
def parse_solver_name(name):
    """Extracts features from the solver configuration name based on configs.py"""
    res = {
        'config_name': name,
        'steps': 2,
        'flow_solver': 'None',
        'pruning': False,
        'packing_solver': 'None',
        'refinement': 'None',
        'delta': False
    }

    # Direct solves
    if name.startswith('flow-'):
        res['steps'] = 1
        res['flow_solver'] = name.split('-')[1]
        res['packing_solver'] = 'Integrated'
        return res

    # 3-step solves (e.g., milp_short-greedy-milp_no_delta)
    if 'no_delta' in name or 'delta' in name:
        res['steps'] = 3
        parts = name.split('-greedy-')
        flow_part = parts[0]
        ref_part = parts[1]

        res['packing_solver'] = 'greedy'
        res['refinement'] = 'milp' if 'milp' in ref_part else 'cp'
        res['delta'] = 'no_delta' not in ref_part
    else:
        # 2-step solves (e.g., cp_short-asp-binpack)
        for p in ['greedy', 'cp-binpack', 'asp-binpack', 'cp']:
            if name.endswith(p):
                res['packing_solver'] = p
                flow_part = name[:-(len(p) + 1)]
                break

    res['pruning'] = 'short' in flow_part
    res['flow_solver'] = flow_part.replace('_short', '')

    return res


def format_solver_display_name(row):
    """Creates a clean, readable name highlighting the step count for the plots."""
    name = row['config_name'].replace('-', ' + ')
    if row['steps'] == 1:
        return f"[1-Step] {name}"
    elif row['steps'] == 2:
        return f"[2-Step] {name}"
    else:
        return f"[3-Step] {name}"


# 2. Load and flatten the data
def load_and_prepare_data(study_name="multibatching-airbus-final_2 copy"):
    with Hdf5Database(f"{study_name}.h5") as database:
        results = database.load_results()

    records = []
    for df in results:
        if len(df) == 0:
            continue  # Skip empty dataframes (timeouts/errors)

        conf_name = df.attrs["config"]["name"]
        inst_name = df.attrs.get("instance", "unknown")

        # Get final objective value (fit) and total time
        final_fit = df["fit"].iloc[-1]
        convergence_time = df.index[-1]

        # Parse features
        features = parse_solver_name(conf_name)
        features.update({
            'instance': inst_name,
            'fitness': final_fit,
            'time_s': convergence_time
        })
        records.append(features)

    df_records = pd.DataFrame(records)
    # Apply the display name formatting
    df_records['display_name'] = df_records.apply(format_solver_display_name, axis=1)

    return df_records


def plot_global_fitness_boxplot(df):
    plt.figure(figsize=(14, 10))

    # Sort by median fitness for a cleaner plot
    order = df.groupby('display_name')['fitness'].median().sort_values().index

    ax = sns.boxplot(
        data=df,
        y='display_name',
        x='fitness',
        order=order,
        palette="vlag"
    )

    plt.title("Distribution of Final Fitness by Solver Configuration", fontsize=14)
    plt.xlabel("Objective Value (Fitness) - Lower is Better", fontsize=12)
    plt.ylabel("Solver Configuration", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("global_fitness_boxplot.pdf")
    plt.show()


def plot_pruning_ablation(df):
    # Filter for 2-step approaches to keep the comparison isolated
    df_2step = df[df['steps'] == 2].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fitness comparison
    sns.boxplot(data=df_2step, x='flow_solver', y='fitness', hue='pruning', ax=axes[0])
    axes[0].set_title("Impact of Shortest-Path Pruning on Fitness")
    axes[0].set_ylabel("Fitness (Lower is Better)")

    # Time comparison
    sns.boxplot(data=df_2step, x='flow_solver', y='time_s', hue='pruning', ax=axes[1])
    axes[1].set_title("Impact of Shortest-Path Pruning on Solve Time")
    axes[1].set_ylabel("Convergence Time (s)")
    axes[1].set_yscale('log')  # Log scale is usually better for time

    plt.tight_layout()
    plt.savefig("pruning_ablation.pdf")
    plt.show()


def plot_delta_ablation(df):
    df_3step = df[df['steps'] == 3].copy()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_3step,
        x='refinement',
        y='fitness',
        hue='delta',
        palette="Set2"
    )
    plt.title("Effect of the Δ-Heuristic on 3-Step Refinement Fitness")
    plt.xlabel("Step 3 Solver (MILP vs CP)")
    plt.ylabel("Fitness")
    plt.tight_layout()
    plt.savefig("delta_ablation.pdf")
    plt.show()


def generate_latex_summary_table(df):
    # Group by the architectural choices
    summary = df.groupby(['steps', 'flow_solver', 'pruning', 'packing_solver', 'refinement']).agg(
        Runs=('fitness', 'count'),
        Mean_Fitness=('fitness', 'mean'),
        Std_Fitness=('fitness', 'std'),
        Median_Time=('time_s', 'median')
    ).round(2)

    # Sort by Mean Fitness to show the best solvers at the top
    summary = summary.sort_values(by='Mean_Fitness')

    # Convert to a LaTeX table
    latex_table = summary.style.to_latex(
        label="tab:solver_performance",
        caption="Performance summary of different solver configurations. Aggregated across all runs and instances.",
        hrules=True,
        environment="table"
    )

    with open("solver_summary.tex", "w") as f:
        f.write(latex_table)

    print(latex_table)


def calculate_relative_gaps(df):
    """Calculates the relative distance to the best solution per instance."""
    df['best_fitness_for_instance'] = df.groupby('instance')['fitness'].transform('min')
    df['relative_gap'] = (df['fitness'] - df['best_fitness_for_instance']) / df['best_fitness_for_instance'].abs()
    df['gap_percentage'] = df['relative_gap'] * 100
    return df


def plot_relative_gap_boxplot(df):
    plt.figure(figsize=(14, 10))

    # Sort solvers by their median gap percentage
    order = df.groupby('display_name')['gap_percentage'].median().sort_values().index

    ax = sns.boxplot(
        data=df,
        y='display_name',
        x='gap_percentage',
        order=order,
        palette="crest",
        showfliers=False  # Hides extreme outliers to make the box shapes clearer
    )

    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.title("Relative Gap to Best Known Solution Across All Instances", fontsize=14)
    plt.xlabel("Gap Percentage (%) - Lower is Better", fontsize=12)
    plt.ylabel("Solver Configuration", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("relative_gap_boxplot.pdf")
    plt.show()


def plot_brick_correlation(df):
    """
    Creates a heatmap correlating the individual bricks (flow solver, packing solver, etc.)
    with the resulting fitness gap and convergence time.
    """
    df_bricks = df.copy()

    # We must ensure we have the gap calculated first
    if 'gap_percentage' not in df_bricks.columns:
        df_bricks = calculate_relative_gaps(df_bricks)

    # One-hot encode the categorical components
    categorical_cols = ['steps', 'flow_solver', 'packing_solver', 'refinement', 'pruning', 'delta']
    df_encoded = pd.get_dummies(df_bricks, columns=categorical_cols)

    # Filter down to just the encoded bricks and the target metrics
    brick_cols = [col for col in df_encoded.columns if any(col.startswith(c) for c in categorical_cols)]
    target_cols = ['gap_percentage', 'time_s']

    # Calculate Spearman correlation
    corr_matrix = df_encoded[brick_cols + target_cols].corr(method='spearman')

    # Isolate the correlations between the Bricks and the Targets
    target_corr = corr_matrix.loc[brick_cols, target_cols].sort_values(by='gap_percentage', ascending=False)

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        target_corr,
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        fmt=".2f",
        cbar_kws={'label': 'Spearman Correlation'}
    )

    plt.title("Impact of Solver Bricks on Performance\n(Negative Gap Correlation = Drives Better Fitness)", fontsize=14)
    plt.ylabel("Solver Components (Bricks)")
    plt.xlabel("Performance Metrics")
    plt.tight_layout()
    plt.savefig("brick_performance_correlation.pdf")
    plt.show()


if __name__ == "__main__":
    # Load all data
    df_full = load_and_prepare_data()

    # Generate Brick Correlation (including all solvers for a complete architectural view)
    print("Generating brick correlation matrix (including all solvers)...")
    plot_brick_correlation(df_full)

    # Filter out 'asp' and 'networkx' for the rest of the analysis as requested
    print("Filtering out ASP and NetworkX for the main analysis...")
    df_filtered = df_full[~df_full['config_name'].str.contains('asp|networkx', case=False, na=False)].copy()

    # Calculate Gaps (on Filtered Data)
    df_filtered_gaps = calculate_relative_gaps(df_filtered)

    # Generate Boxplots & Ablations (on Filtered Data)
    plot_global_fitness_boxplot(df_filtered_gaps)
    plot_relative_gap_boxplot(df_filtered_gaps)
    plot_pruning_ablation(df_filtered_gaps)
    plot_delta_ablation(df_filtered_gaps)

    # Generate LaTeX Summary Tables
    generate_latex_summary_table(df_filtered_gaps)