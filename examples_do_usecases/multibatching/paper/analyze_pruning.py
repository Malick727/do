import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from discrete_optimization.generic_tools.study import Hdf5Database


def load_and_interpolate_timeseries(study_name="multibatching-airbus-shortest-path", max_time=200, step=1):
    """
    Loads timeseries data, computes exact per-run statistics (like time to first solution),
    and interpolates it onto a uniform time grid for aggregated plotting.
    """
    with Hdf5Database(f"{study_name}.h5") as database:
        results = database.load_results()

    grid = np.arange(0, max_time + step, step)
    all_data = []
    run_stats = []

    for df in results:
        if len(df) == 0:
            continue

        conf_name = df.attrs["config"]["name"]
        inst_name = df.attrs.get("instance", "unknown")

        solver_type = 'milp' if 'milp' in conf_name else 'cp'
        pruning = 'True' if 'short' in conf_name else 'False'

        # Normalize time index to seconds
        if isinstance(df.index, pd.TimedeltaIndex):
            t_sec = df.index.total_seconds()
        elif isinstance(df.index, pd.DatetimeIndex):
            t_sec = (df.index - df.index[0]).total_seconds()
        else:
            t_sec = df.index.astype(float)

        # Safely extract obj and bound
        obj = df.get('obj', df.get('fit', pd.Series(np.nan, index=df.index)))
        bound = df.get('bound', pd.Series(np.nan, index=df.index))

        temp_df = pd.DataFrame({'time_s': t_sec, 'obj': obj.values, 'bound': bound.values})

        # Remove negative lower bounds (dummy -infinity before root node is solved)
        temp_df.loc[temp_df['bound'] < 0, 'bound'] = np.nan

        # Drop duplicate timestamps keeping the most recent logged value
        temp_df = temp_df.drop_duplicates(subset='time_s', keep='last')
        temp_df.set_index('time_s', inplace=True)

        # --- Extract Time to First Solution (Before Interpolation) ---
        valid_objs = temp_df.loc[(temp_df['obj'].notna()) & (temp_df['obj'] < 1e12)]
        first_sol_time = valid_objs.index[0] if not valid_objs.empty else np.nan

        run_stats.append({
            'solver': solver_type,
            'pruning': pruning,
            'run': inst_name,
            'first_sol_time': first_sol_time
        })

        # --- Interpolation ---
        # Union the actual logged times with our standard grid
        union_idx = np.union1d(grid, temp_df.index)

        # Forward-fill missing points
        temp_df = temp_df.reindex(union_idx).ffill()

        # Keep only the uniform grid points
        grid_df = temp_df.loc[grid].reset_index().rename(columns={'index': 'time_s'})

        grid_df['config'] = conf_name
        grid_df['solver'] = solver_type
        grid_df['pruning'] = pruning
        grid_df['run'] = inst_name

        all_data.append(grid_df)

    df_final = pd.concat(all_data, ignore_index=True)
    df_stats = pd.DataFrame(run_stats)

    # Remove CP-SAT's dummy initial objectives so they don't break the plot scales
    global_best_obj = df_final['obj'].min()
    df_final.loc[df_final['obj'] > (global_best_obj * 3), 'obj'] = np.nan

    # Calculate Absolute Gap
    df_final['abs_gap'] = df_final['obj'] - df_final['bound']
    df_final['abs_gap'] = df_final['abs_gap'].clip(lower=1)  # Clip at 1 to prevent log(0) errors

    return df_final, df_stats


def plot_isolated_convergence(df, solver_name, include_bound=True):
    """
    Creates a standalone plot for a specific solver.
    """
    df_solver = df[df['solver'] == solver_name].copy()

    plt.figure(figsize=(8, 6))

    if include_bound:
        df_melted = df_solver.melt(id_vars=['time_s', 'pruning', 'run', 'config'],
                                   value_vars=['obj', 'bound'],
                                   var_name='metric', value_name='value')

        sns.lineplot(data=df_melted, x="time_s", y="value", hue="pruning", style="metric", errorbar='sd', linewidth=2)
        title = f"{solver_name.upper()} convergence"
        filename = f"step1_{solver_name}_convergence.pdf"
    else:
        sns.lineplot(data=df_solver, x="time_s", y="obj", hue="pruning", errorbar='sd', linewidth=2)
        title = f"{solver_name.upper()} objective convergence"
        filename = f"step1_{solver_name}_obj_convergence.pdf"

    plt.yscale('log')
    plt.title(title, fontsize=14)
    plt.ylabel("Objective / Bound", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_solver_comparison_baseline(df):
    """
    Directly compares the objective convergence of MILP vs CP,
    strictly for the baseline runs where no pruning is applied.
    """
    # Filter for the baseline configuration (no pruning)
    df_baseline = df[df['pruning'] == 'False'].copy()

    plt.figure(figsize=(9, 6))

    # Using errorbar=None to keep the lines clean and focused on the average trend
    sns.lineplot(
        data=df_baseline,
        x='time_s',
        y='obj',
        hue='solver',
        errorbar=None,
        linewidth=2.5,
        palette='Set1'
    )

    #plt.yscale('log')
    plt.title("Convergence Speed: MILP vs CP", fontsize=14)
    plt.ylabel("Average Objective Function", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)

    # Improve the legend readability
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=[l.upper() for l in labels], title="Flow Solver")

    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("step1_milp_vs_cp_baseline.pdf")
    plt.show()


def plot_gap_evolution(df):
    """
    Plots the convergence of the absolute optimality gap.
    """
    plt.figure(figsize=(8, 6))

    sns.lineplot(data=df, x='time_s', y='abs_gap', hue='solver', style='pruning', errorbar='sd', palette='tab10',
                 linewidth=2)

    plt.yscale('log')
    plt.title("Absolute gap evolution", fontsize=14)
    plt.ylabel("Absolute gap", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("step1_gap_evolution.pdf")
    plt.show()


def generate_convergence_latex_table(df_timeseries, df_stats, filename="step1_convergence_summary.tex"):
    """
    Generates a cleanly formatted LaTeX table analyzing the final state and time to first solution.
    """
    max_t = df_timeseries['time_s'].max()
    df_final = df_timeseries[df_timeseries['time_s'] == max_t].copy()

    df_merged = pd.merge(df_final, df_stats, on=['solver', 'pruning', 'run'])

    summary = df_merged.groupby(['solver', 'pruning']).agg(
        Runs=('run', 'nunique'),
        Mean_Obj=('obj', 'mean'),
        Std_Obj=('obj', 'std'),
        Mean_Bound=('bound', 'mean'),
        Std_Bound=('bound', 'std'),
        Mean_Abs_Gap=('abs_gap', 'mean'),
        Mean_T1=('first_sol_time', 'mean')
    ).reset_index()

    summary = summary.sort_values(by=['solver', 'pruning'])

    min_obj = summary['Mean_Obj'].min()
    max_bound = summary['Mean_Bound'].max()

    summary['Runs'] = summary['Runs'].astype(int).astype(str)

    summary['Mean_Obj'] = summary['Mean_Obj'].apply(
        lambda x: f"\\textbf{{{x:.0f}}}" if pd.notnull(x) and x == min_obj else (f"{x:.0f}" if pd.notnull(x) else "-")
    )
    summary['Std_Obj'] = summary['Std_Obj'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "-")

    summary['Mean_Bound'] = summary['Mean_Bound'].apply(
        lambda x: f"\\textbf{{{x:.0f}}}" if pd.notnull(x) and x == max_bound else (f"{x:.0f}" if pd.notnull(x) else "-")
    )
    summary['Std_Bound'] = summary['Std_Bound'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "-")

    summary['Mean_Abs_Gap'] = summary['Mean_Abs_Gap'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "-")

    summary['Mean_T1'] = summary['Mean_T1'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "-")

    summary = summary.rename(columns={
        'solver': 'Flow Solver',
        'pruning': 'Pruning',
        'Mean_Obj': 'Final Mean Obj',
        'Std_Obj': 'Std Obj',
        'Mean_Bound': 'Final Mean Bound',
        'Std_Bound': 'Std Bound',
        'Mean_Abs_Gap': 'Final Abs Gap',
        'Mean_T1': 'Time to 1st Sol (s)'
    })

    summary = summary.set_index(['Flow Solver', 'Pruning'])

    latex_table = summary.to_latex(
        label="tab:step1_convergence",
        caption="Flow routing convergence summary at 200 seconds. Best mean objective and lower bound are highlighted in bold.",
        escape=False,
        multirow=True
    )

    with open(filename, "w") as f:
        f.write(latex_table)
    print(f"-> Generated {filename}")


if __name__ == "__main__":
    print("Loading and interpolating time series data...")
    df_timeseries, df_stats = load_and_interpolate_timeseries(study_name="multibatching-airbus-shortest-path-2",
                                                              max_time=200)

    # 1. Generate the new direct comparison plot (MILP vs CP)
    print("Generating baseline MILP vs CP comparison plot...")
    plot_solver_comparison_baseline(df_timeseries)

    # 2. Generate separated MILP plots
    print("Generating MILP plots...")
    plot_isolated_convergence(df_timeseries, solver_name='milp', include_bound=True)
    plot_isolated_convergence(df_timeseries, solver_name='milp', include_bound=False)

    # 3. Generate separated CP plots
    print("Generating CP plots...")
    plot_isolated_convergence(df_timeseries, solver_name='cp', include_bound=True)
    plot_isolated_convergence(df_timeseries, solver_name='cp', include_bound=False)

    # 4. Generate gap and table
    print("Generating gap evolution plot...")
    plot_gap_evolution(df_timeseries)

    print("Generating LaTeX summary table...")
    generate_convergence_latex_table(df_timeseries, df_stats)

    print("Analysis complete. All plots will display until closed.")