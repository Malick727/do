"""
Interactive Dashboard for Engine Preparation and Allocation solutions

Features:
- Time-based tracking with slider
- Current state at any time point:
  * Engines in preparation at each factory
  * Engines in transit to FALs
  * Engines arrived at FALs
- Cumulative KPIs over time
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    Engine,
    MSN,
    PreparationFactory,
    AssemblyLine,
)


# Color schemes
ENGINE_TYPE_COLORS = {
    "TrentXWB": "#3498db",  # Bright Blue
    "GE90": "#e74c3c",      # Bright Red
    "T1": "#2ecc71",        # Green
}

FACTORY_COLORS = {
    "Derby_UK": "#9b59b6",
    "Toulouse_FR": "#1abc9c",
    "Virginia_US": "#f39c12",
    "Singapore_SG": "#e67e22",
    "Durham_UK": "#34495e",
    "Bordeaux_FR": "#16a085",
    "Cincinnati_US": "#d35400",
    "Shanghai_CN": "#c0392b",
}

FAL_COLORS = {
    "FAL_Hamburg": "#8e44ad",
    "FAL_Toulouse": "#27ae60",
    "FAL_Mobile": "#2980b9",
    "FAL_Tianjin": "#c0392b",
    "FAL1": "#7f8c8d",
}


@dataclass
class TimeSnapshot:
    """State of the system at a specific time"""
    time: int
    engines_in_prep: Dict[str, List[str]]  # factory_id -> [engine_ids]
    engines_in_transit: List[Tuple[str, str, str]]  # [(engine_id, factory_id, fal_id)]
    engines_arrived: Dict[str, List[Tuple[str, int]]]  # fal_id -> [(engine_id, arrival_time)]
    cumulative_delays: float
    cumulative_cost: float
    baseline_changes: int


def get_fal_proximity_groups(problem: EnginePreparationProblem) -> Dict[str, List[PreparationFactory]]:
    """
    Group factories by their closest FAL based on travel time/cost

    Returns:
        Dict mapping FAL_id -> list of factories close to it
    """
    fal_to_factories = defaultdict(list)

    for factory in problem.factories_list:
        # Find closest FAL to this factory
        min_cost = float('inf')
        closest_fal = None

        for fal in problem.assembly_lines_list:
            cost = problem.get_travel_cost(factory, fal)
            time = problem.get_travel_time(factory, fal)
            # Use weighted combination
            metric = cost + time * 100

            if metric < min_cost:
                min_cost = metric
                closest_fal = fal.id

        if closest_fal:
            fal_to_factories[closest_fal].append(factory)

    return fal_to_factories


def create_gantt_chart(
    problem: EnginePreparationProblem,
    solution: EnginePreparationSolution,
    title: str = "Engine Preparation Timeline by Factory"
) -> go.Figure:
    """
    Create Gantt chart showing engine preparation at each factory
    Organized by FAL proximity
    """
    fal_groups = get_fal_proximity_groups(problem)

    # Prepare data for Gantt chart
    gantt_data = []

    # Sort FALs by ID for consistent ordering
    sorted_fals = sorted(fal_groups.keys())

    for fal_id in sorted_fals:
        factories = fal_groups[fal_id]

        for factory in sorted(factories, key=lambda f: f.id):
            # Find all engines processed at this factory
            engines_at_factory = [
                (engine, solution.prep_start_times[engine], solution.prep_end_times[engine])
                for engine in problem.engines_list
                if engine in solution.engine_to_factory and solution.engine_to_factory[engine] == factory
            ]

            # Sort by start time
            engines_at_factory.sort(key=lambda x: x[1])

            for engine, start, end in engines_at_factory:
                msn = solution.engine_to_msn.get(engine)
                arrival = solution.fal_arrival_times.get(engine)

                # Check for delay
                delay = 0
                if msn and arrival:
                    delay = max(0, arrival - msn.engine_installation_date)

                gantt_data.append({
                    'Task': f"{factory.id} (→{fal_id})",
                    'Start': start,
                    'Finish': end,
                    'Engine': engine.id,
                    'EngineType': engine.engine_type,
                    'MSN': msn.id if msn else "?",
                    'FAL': fal_id,
                    'Delay': f"{delay:.0f} days" if delay > 0 else "On time",
                    'Factory': factory.id,
                })

    df = pd.DataFrame(gantt_data)

    # Create figure
    fig = go.Figure()

    # Add trace for each engine type
    for engine_type in df['EngineType'].unique():
        df_type = df[df['EngineType'] == engine_type]

        for idx, row in df_type.iterrows():
            fig.add_trace(go.Bar(
                name=engine_type,
                x=[row['Finish'] - row['Start']],
                y=[row['Task']],
                base=[row['Start']],
                orientation='h',
                marker=dict(
                    color=ENGINE_TYPE_COLORS.get(engine_type, '#cccccc'),
                    line=dict(color='white', width=1)
                ),
                text=row['Engine'],
                textposition='inside',
                hovertemplate=(
                    f"<b>{row['Engine']}</b><br>"
                    f"Type: {row['EngineType']}<br>"
                    f"MSN: {row['MSN']}<br>"
                    f"Factory: {row['Factory']}<br>"
                    f"Prep: {row['Start']:.0f} → {row['Finish']:.0f}<br>"
                    f"Status: {row['Delay']}<br>"
                    "<extra></extra>"
                ),
                showlegend=bool(idx == df_type.index[0]),  # Only show legend once per type
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (days)",
        yaxis_title="Preparation Factory (→ Closest FAL)",
        barmode='overlay',
        height=max(400, len(df['Task'].unique()) * 30),
        hovermode='closest',
        yaxis=dict(autorange="reversed"),  # Top to bottom
    )

    return fig


def create_factory_workload_chart(
    problem: EnginePreparationProblem,
    solution: EnginePreparationSolution,
    title: str = "Factory Workload Over Time"
) -> go.Figure:
    """
    Show workload (number of concurrent engines) at each factory over time
    """
    fig = make_subplots(
        rows=len(problem.factories_list),
        cols=1,
        subplot_titles=[f"{f.id} (capacity: {f.capacity if f.capacity else '∞'}/week)"
                       for f in sorted(problem.factories_list, key=lambda x: x.id)],
        vertical_spacing=0.03,
    )

    for idx, factory in enumerate(sorted(problem.factories_list, key=lambda x: x.id), 1):
        # Calculate workload over time
        time_points = []
        workload = []

        # Get all engines at this factory
        engines_at_factory = [
            (engine, solution.prep_start_times[engine], solution.prep_end_times[engine])
            for engine in problem.engines_list
            if engine in solution.engine_to_factory and solution.engine_to_factory[engine] == factory
        ]

        if not engines_at_factory:
            continue

        # Sample time points
        for t in range(0, problem.time_horizon + 1):
            # Count concurrent engines
            count = sum(
                1 for engine, start, end in engines_at_factory
                if start <= t < end
            )
            time_points.append(t)
            workload.append(count)

        # Add workload trace
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=workload,
                fill='tozeroy',
                name=factory.id,
                line=dict(color='steelblue', width=2),
                hovertemplate="Time: %{x}<br>Workload: %{y} engines<extra></extra>",
            ),
            row=idx,
            col=1
        )

        # Add capacity line if specified
        if factory.capacity is not None:
            fig.add_trace(
                go.Scatter(
                    x=[0, problem.time_horizon],
                    y=[factory.capacity, factory.capacity],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name=f"Capacity ({factory.capacity})",
                    hovertemplate=f"Capacity: {factory.capacity}<extra></extra>",
                ),
                row=idx,
                col=1
            )

        # Update y-axis
        fig.update_yaxes(title_text="Engines", row=idx, col=1)

    fig.update_xaxes(title_text="Time (days)", row=len(problem.factories_list), col=1)
    fig.update_layout(
        title=title,
        height=max(600, len(problem.factories_list) * 150),
        showlegend=False,
    )

    return fig


def create_fal_arrival_timeline(
    problem: EnginePreparationProblem,
    solution: EnginePreparationSolution,
    title: str = "Engine Arrivals at Final Assembly Lines"
) -> go.Figure:
    """
    Show engine arrivals at each FAL with delays highlighted
    """
    fig = make_subplots(
        rows=len(problem.assembly_lines_list),
        cols=1,
        subplot_titles=[f"{fal.id} ({fal.location})" for fal in sorted(problem.assembly_lines_list, key=lambda x: x.id)],
        vertical_spacing=0.05,
    )

    for idx, fal in enumerate(sorted(problem.assembly_lines_list, key=lambda x: x.id), 1):
        # Get all MSNs at this FAL
        msns_at_fal = [m for m in problem.msns_list if m.assembly_line_id == fal.id]

        # Get arrivals
        arrivals = []
        for msn in msns_at_fal:
            # Find engine assigned to this MSN
            engine = next((e for e in problem.engines_list
                          if e in solution.engine_to_msn and solution.engine_to_msn[e] == msn), None)

            if engine and engine in solution.fal_arrival_times:
                arrival_time = solution.fal_arrival_times[engine]
                deadline = msn.engine_installation_date
                delay = max(0, arrival_time - deadline)

                arrivals.append({
                    'MSN': msn.id,
                    'Engine': engine.id,
                    'EngineType': engine.engine_type,
                    'Arrival': arrival_time,
                    'Deadline': deadline,
                    'Delay': delay,
                    'OnTime': delay == 0,
                })

        if not arrivals:
            continue

        df_arrivals = pd.DataFrame(arrivals)
        df_arrivals = df_arrivals.sort_values('Arrival')

        # Plot arrivals vs deadlines
        for engine_type in df_arrivals['EngineType'].unique():
            df_type = df_arrivals[df_arrivals['EngineType'] == engine_type]

            # On-time arrivals
            df_ontime = df_type[df_type['OnTime']]
            if not df_ontime.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_ontime['Arrival'],
                        y=df_ontime['MSN'],
                        mode='markers',
                        name=f"{engine_type} (on-time)",
                        marker=dict(
                            color=ENGINE_TYPE_COLORS.get(engine_type, '#cccccc'),
                            size=10,
                            symbol='circle',
                        ),
                        hovertemplate=(
                            "MSN: %{y}<br>"
                            "Engine: %{customdata[0]}<br>"
                            "Arrival: %{x}<br>"
                            "Deadline: %{customdata[1]}<br>"
                            "Status: On time<br>"
                            "<extra></extra>"
                        ),
                        customdata=df_ontime[['Engine', 'Deadline']].values,
                    ),
                    row=idx,
                    col=1
                )

            # Late arrivals
            df_late = df_type[~df_type['OnTime']]
            if not df_late.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_late['Arrival'],
                        y=df_late['MSN'],
                        mode='markers',
                        name=f"{engine_type} (LATE)",
                        marker=dict(
                            color='red',
                            size=12,
                            symbol='x',
                            line=dict(width=2),
                        ),
                        hovertemplate=(
                            "MSN: %{y}<br>"
                            "Engine: %{customdata[0]}<br>"
                            "Arrival: %{x}<br>"
                            "Deadline: %{customdata[1]}<br>"
                            "Delay: %{customdata[2]:.0f} days<br>"
                            "<extra></extra>"
                        ),
                        customdata=df_late[['Engine', 'Deadline', 'Delay']].values,
                    ),
                    row=idx,
                    col=1
                )

        # Add deadline markers
        fig.add_trace(
            go.Scatter(
                x=df_arrivals['Deadline'],
                y=df_arrivals['MSN'],
                mode='markers',
                name='Deadline',
                marker=dict(
                    color='black',
                    size=8,
                    symbol='line-ns',
                    line=dict(width=2),
                ),
                hovertemplate="Deadline: %{x}<extra></extra>",
            ),
            row=idx,
            col=1
        )

        fig.update_yaxes(title_text="MSN", row=idx, col=1)

    fig.update_xaxes(title_text="Time (days)", row=len(problem.assembly_lines_list), col=1)
    fig.update_layout(
        title=title,
        height=max(600, len(problem.assembly_lines_list) * 200),
        showlegend=True,
    )

    return fig


def create_cumulative_delay_chart(
    problem: EnginePreparationProblem,
    solution: EnginePreparationSolution,
    title: str = "Cumulative Delays at FALs Over Time"
) -> go.Figure:
    """
    Show cumulative delays accumulating at each FAL over time
    """
    fig = go.Figure()

    for fal in sorted(problem.assembly_lines_list, key=lambda x: x.id):
        # Get all MSNs at this FAL
        msns_at_fal = [m for m in problem.msns_list if m.assembly_line_id == fal.id]

        # Calculate cumulative delays over time
        time_delay_events = []

        for msn in msns_at_fal:
            engine = next((e for e in problem.engines_list
                          if e in solution.engine_to_msn and solution.engine_to_msn[e] == msn), None)

            if engine and engine in solution.fal_arrival_times:
                arrival_time = solution.fal_arrival_times[engine]
                deadline = msn.engine_installation_date
                delay = max(0, arrival_time - deadline)

                time_delay_events.append((deadline, delay))

        # Sort by deadline
        time_delay_events.sort()

        # Calculate cumulative
        times = [0]
        cumulative_delays = [0]

        for deadline, delay in time_delay_events:
            times.append(deadline)
            cumulative_delays.append(cumulative_delays[-1] + delay)

        fig.add_trace(
            go.Scatter(
                x=times,
                y=cumulative_delays,
                mode='lines+markers',
                name=fal.id,
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate=(
                    f"<b>{fal.id}</b><br>"
                    "Time: %{x}<br>"
                    "Cumulative Delay: %{y:.0f} days<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (days)",
        yaxis_title="Cumulative Delay (days)",
        height=400,
        hovermode='x unified',
    )

    return fig


def visualize_solution(
    problem: EnginePreparationProblem,
    solution: EnginePreparationSolution,
    save_html: str = None,
) -> None:
    """
    Create comprehensive interactive visualization of the solution

    Args:
        problem: Problem instance
        solution: Solution to visualize
        save_html: Optional path to save HTML file
    """
    from plotly.subplots import make_subplots

    print("Creating visualizations...")

    # Create individual charts
    gantt_fig = create_gantt_chart(problem, solution)
    workload_fig = create_factory_workload_chart(problem, solution)
    arrival_fig = create_fal_arrival_timeline(problem, solution)
    cumulative_fig = create_cumulative_delay_chart(problem, solution)

    # Show all charts
    gantt_fig.show()
    workload_fig.show()
    arrival_fig.show()
    cumulative_fig.show()

    # Optionally save to HTML
    if save_html:
        # Combine all into one HTML with tabs (using basic HTML structure)
        with open(save_html, 'w') as f:
            f.write("""
            <html>
            <head>
                <title>Engine Preparation Solution Visualization</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    .chart { margin-bottom: 40px; border: 1px solid #ddd; padding: 10px; }
                </style>
            </head>
            <body>
                <h1>Engine Preparation and Allocation - Solution Visualization</h1>
            """)

            f.write('<div class="chart">')
            f.write(gantt_fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('</div>')

            f.write('<div class="chart">')
            f.write(workload_fig.to_html(full_html=False, include_plotlyjs=False))
            f.write('</div>')

            f.write('<div class="chart">')
            f.write(arrival_fig.to_html(full_html=False, include_plotlyjs=False))
            f.write('</div>')

            f.write('<div class="chart">')
            f.write(cumulative_fig.to_html(full_html=False, include_plotlyjs=False))
            f.write('</div>')

            f.write("""
            </body>
            </html>
            """)

        print(f"\n✓ Visualization saved to: {save_html}")