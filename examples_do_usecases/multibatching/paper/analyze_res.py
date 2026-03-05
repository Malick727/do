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
def load_and_prepare_data(study_name="multibatching-airbus-final_4xps"):
    results = []
    with Hdf5Database(f"{study_name}.h5") as database:
        results = database.load_results()
        results = [r for r in results
                   if r.attrs["config"]["name"][:3]!="asp"]
    with Hdf5Database("multibatching-airbus-asp.h5") as database:
        results.extend(database.load_results())


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


def calculate_relative_gaps(df):
    """Calculates the relative distance to the best solution per instance."""
    df['best_fitness_for_instance'] = df.groupby('instance')['fitness'].transform('min')
    df['relative_gap'] = (df['fitness'] - df['best_fitness_for_instance']) / df['best_fitness_for_instance'].abs()
    df['gap_percentage'] = df['relative_gap'] * 100
    return df


def plot_global_fitness_boxplot(df):
    plt.figure(figsize=(14, 10))
    order = df.groupby('display_name')['fitness'].median().sort_values().index
    sns.boxplot(data=df, y='display_name', x='fitness', order=order, palette="vlag")
    plt.title("Distribution of Final Fitness by Solver Configuration", fontsize=14)
    plt.xlabel("Objective Value (Fitness) - Lower is Better", fontsize=12)
    plt.ylabel("Solver Configuration", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("global_fitness_boxplot.pdf")
    plt.show()


def plot_relative_gap_boxplot(df):
    plt.figure(figsize=(14, 10))
    order = df.groupby('display_name')['gap_percentage'].median().sort_values().index
    sns.boxplot(
        data=df, y='display_name', x='gap_percentage', order=order, palette="crest", showfliers=False
    )
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.title("Relative Gap to Best Known Solution Across All Instances", fontsize=14)
    plt.xlabel("Gap Percentage (%) - Lower is Better", fontsize=12)
    plt.ylabel("Solver Configuration", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("relative_gap_boxplot.pdf")
    plt.show()


def plot_pruning_ablation(df):
    # Strictly filtered to 2-step approaches to avoid bias
    df_2step = df[df['steps'] == 2].copy()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_2step, x='flow_solver', y='fitness', hue='pruning')
    plt.title("Impact of Shortest-Path Pruning on Fitness (2-Step Only)")
    plt.ylabel("Fitness (Lower is Better)")
    plt.xlabel("Flow Solver")

    plt.tight_layout()
    plt.savefig("pruning_ablation_fitness.pdf")
    plt.show()


def plot_delta_ablation(df):
    df_3step = df[df['steps'] == 3].copy()
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_3step, x='refinement', y='fitness', hue='delta', palette="Set2")
    plt.title("Effect of the $\\Delta$-Heuristic on 3-Step Refinement Fitness")
    plt.xlabel("Step 3 Solver (MILP vs CP)")
    plt.ylabel("Fitness")
    plt.tight_layout()
    plt.savefig("delta_ablation.pdf")
    plt.show()


def plot_brick_correlation_all(df):
    df_bricks = df.copy()
    if 'gap_percentage' not in df_bricks.columns:
        df_bricks = calculate_relative_gaps(df_bricks)

    categorical_cols = ['steps', 'flow_solver', 'packing_solver', 'refinement', 'pruning', 'delta']
    df_encoded = pd.get_dummies(df_bricks, columns=categorical_cols)

    brick_cols = [col for col in df_encoded.columns if any(col.startswith(c) for c in categorical_cols)]
    target_cols = ['gap_percentage']

    corr_matrix = df_encoded[brick_cols + target_cols].corr(method='spearman')
    target_corr = corr_matrix.loc[brick_cols, target_cols].sort_values(by='gap_percentage', ascending=False)

    plt.figure(figsize=(6, 10))
    sns.heatmap(
        target_corr,
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        fmt=".2f",
        cbar_kws={'label': 'Spearman Correlation'}
    )

    plt.title("Impact of Solver Bricks on Fitness Gap (All Steps)\n(Negative values = Reduces gap/Improves fitness)",
              fontsize=11)
    plt.ylabel("Solver Components (Bricks)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("brick_performance_correlation_all.pdf")
    plt.show()


def plot_brick_correlation_2step(df):
    df_bricks = df[df['steps'] == 2].copy()
    if 'gap_percentage' not in df_bricks.columns:
        df_bricks = calculate_relative_gaps(df_bricks)

    categorical_cols = ['flow_solver', 'packing_solver', 'pruning']
    df_encoded = pd.get_dummies(df_bricks, columns=categorical_cols)

    brick_cols = [col for col in df_encoded.columns if any(col.startswith(c) for c in categorical_cols)]
    target_cols = ['gap_percentage']

    corr_matrix = df_encoded[brick_cols + target_cols].corr(method='spearman')
    target_corr = corr_matrix.loc[brick_cols, target_cols].sort_values(by='gap_percentage', ascending=False)

    plt.figure(figsize=(6, 8))
    sns.heatmap(
        target_corr,
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        fmt=".2f",
        cbar_kws={'label': 'Spearman Correlation'}
    )

    plt.title("Impact of Solver Bricks on Fitness Gap (2-Step Only)\n(Negative values = Reduces gap/Improves fitness)",
              fontsize=11)
    plt.ylabel("Solver Components (Bricks)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("brick_performance_correlation_2step.pdf")
    plt.show()


def generate_latex_summary_table(df, filename="solver_summary.tex"):
    summary = df.groupby(['steps', 'flow_solver', 'pruning', 'packing_solver', 'refinement', 'delta']).agg(
        Runs=('fitness', 'count'),
        Mean_Fitness=('fitness', 'mean'),
        Std_Fitness=('fitness', 'std')
    ).reset_index()

    # Combine refinement and delta logic cleanly for the table
    def format_refinement(row):
        if row['steps'] == 3:
            delta_str = " (with $\\Delta$)" if row['delta'] else " (no $\\Delta$)"
            return f"{row['refinement']}{delta_str}"
        return row['refinement']

    summary['refinement'] = summary.apply(format_refinement, axis=1)
    summary = summary.drop(columns=['delta'])

    # Rename columns to remove underscores in the header
    summary = summary.rename(columns={
        'steps': 'Steps',
        'flow_solver': 'Flow Solver',
        'pruning': 'Pruning',
        'packing_solver': 'Packing Solver',
        'refinement': 'Refinement',
        'Mean_Fitness': 'Mean Fitness',
        'Std_Fitness': 'Std Fitness'
    })

    # Escape any underscores in the solver component names
    for col in ['Flow Solver', 'Packing Solver', 'Refinement']:
        summary[col] = summary[col].astype(str).str.replace('_', '\\_')

    # THIS IS THE VITAL FIX: Sort lexicographically by parameters BEFORE setting index
    summary = summary.sort_values(by=['Steps', 'Flow Solver', 'Pruning', 'Packing Solver', 'Refinement'])

    # Find the absolute best mean fitness to bold it
    min_fit = summary['Mean Fitness'].min()

    # Format strings manually so we can insert \textbf
    summary['Runs'] = summary['Runs'].astype(int).astype(str)
    summary['Mean Fitness'] = summary['Mean Fitness'].apply(
        lambda x: f"\\textbf{{{x:.0f}}}" if pd.notnull(x) and x == min_fit else (f"{x:.0f}" if pd.notnull(x) else "")
    )
    summary['Std Fitness'] = summary['Std Fitness'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "")

    # Set all categorical parameters as the MultiIndex to trigger multirow
    summary = summary.set_index(['Steps', 'Flow Solver', 'Pruning', 'Packing Solver', 'Refinement'])

    latex_table = summary.to_latex(
        label="tab:solver_performance",
        caption="Performance summary grouped lexicographically by parameters. The best mean fitness is highlighted in bold.",
        escape=False,
        multirow=True
    )

    with open(filename, "w") as f:
        f.write(latex_table)
    print(f"-> Generated {filename}")


def generate_latex_gap_table(df, filename="solver_gap_summary.tex"):
    df['is_best'] = df['gap_percentage'] == 0.0

    # Group by the broken-out parameters instead of config_name for a cleaner table!
    summary = df.groupby(['steps', 'flow_solver', 'pruning', 'packing_solver', 'refinement', 'delta']).agg(
        Total_Runs=('fitness', 'count'),
        Best_Found=('is_best', 'sum'),
        Mean_Gap_Pct=('gap_percentage', 'mean'),
        Median_Gap_Pct=('gap_percentage', 'median'),
        Max_Gap_Pct=('gap_percentage', 'max')
    ).reset_index()

    # Format Refinement column
    def format_refinement(row):
        if row['steps'] == 3:
            delta_str = " (with $\\Delta$)" if row['delta'] else " (no $\\Delta$)"
            return f"{row['refinement']}{delta_str}"
        return row['refinement']

    summary['refinement'] = summary.apply(format_refinement, axis=1)
    summary = summary.drop(columns=['delta'])

    # Clean headers
    summary = summary.rename(columns={
        'steps': 'Steps',
        'flow_solver': 'Flow Solver',
        'pruning': 'Pruning',
        'packing_solver': 'Packing Solver',
        'refinement': 'Refinement',
        'Total_Runs': 'Total Runs',
        'Best_Found': 'Best Found',
        'Mean_Gap_Pct': 'Mean Gap (\\%)',
        'Median_Gap_Pct': 'Median Gap (\\%)',
        'Max_Gap_Pct': 'Max Gap (\\%)'
    })

    # Escape underscores in the config names
    for col in ['Flow Solver', 'Packing Solver', 'Refinement']:
        summary[col] = summary[col].astype(str).str.replace('_', '\\_')

    # SORT LEXICOGRAPHICALLY to trigger perfect multirows
    summary = summary.sort_values(by=['Steps', 'Flow Solver', 'Pruning', 'Packing Solver', 'Refinement'])

    min_gap = summary['Mean Gap (\\%)'].min()
    max_best = summary['Best Found'].max()

    # Apply formatting and bolding
    summary['Total Runs'] = summary['Total Runs'].astype(int).astype(str)
    summary['Best Found'] = summary['Best Found'].apply(
        lambda x: f"\\textbf{{{x}}}" if x == max_best and x > 0 else str(x)
    )
    for col in ['Mean Gap (\\%)', 'Median Gap (\\%)', 'Max Gap (\\%)']:
        summary[col] = summary[col].apply(
            lambda x: f"\\textbf{{{x:.0f}}}" if (col == 'Mean Gap (\\%)' and pd.notnull(x) and x == min_gap) else (
                f"{x:.0f}" if pd.notnull(x) else "")
        )

    # Set index to trigger multirow
    summary = summary.set_index(['Steps', 'Flow Solver', 'Pruning', 'Packing Solver', 'Refinement'])

    latex_table = summary.to_latex(
        label="tab:relative_gap_performance",
        caption="Relative gap percentage to the BKS grouped hierarchically. Best results are in bold.",
        escape=False,
        multirow=True
    )

    with open(filename, "w") as f:
        f.write(latex_table)
    print(f"-> Generated {filename}")


if __name__ == "__main__":
    # 1. Load data and calculate gaps across ALL configurations
    df_full = load_and_prepare_data()
    df_full_gaps = calculate_relative_gaps(df_full)

    # 2. Generate Brick Correlations
    print("Generating brick correlation matrix (All Steps)...")
    plot_brick_correlation_all(df_full_gaps)

    print("Generating brick correlation matrix (2-Step Only)...")
    plot_brick_correlation_2step(df_full_gaps)

    # 3. Filter out NetworkX for tables and plots
    df_no_nx = df_full_gaps[~df_full_gaps['config_name'].str.contains('networkx', case=False, na=False)].copy()

    # 4. Filter out ASP just for the plots (to prevent visual skew on the axes)
    print("Generating Boxplots & Ablations...")
    df_plots = df_no_nx[~df_no_nx['config_name'].str.contains('asp', case=False, na=False)].copy()

    plot_global_fitness_boxplot(df_plots)
    plot_relative_gap_boxplot(df_plots)
    plot_pruning_ablation(df_plots)
    plot_delta_ablation(df_plots)

    # 5. Generate LaTeX Summary Tables (Includes ASP, strictly grouped)
    print("Generating LaTeX tables (hierarchically grouped)...")
    generate_latex_summary_table(df_no_nx)
    generate_latex_gap_table(df_no_nx)

    print("Analysis complete. All plots will display until closed.")