"""
Time-based Dashboard for Engine Preparation and Allocation

Interactive dashboard that tracks system state over time with animated charts
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

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
    "TrentXWB": "#3498db",
    "GE90": "#e74c3c",
    "T1": "#2ecc71",
}

FACTORY_BASE_COLORS = [
    "#9b59b6", "#3498db", "#e74c3c", "#f39c12",
    "#1abc9c", "#2ecc71", "#e67e22", "#34495e"
]


def compute_timeline_data(problem: EnginePreparationProblem, solution: EnginePreparationSolution):
    """
    Compute time-series data for visualization

    Returns dict with timeline info for each time step
    """
    timeline = {}

    for t in range(0, problem.time_horizon + 1):
        # Engines currently in preparation
        engines_in_prep = defaultdict(list)
        for engine in problem.engines_list:
            if engine in solution.prep_start_times:
                prep_start = solution.prep_start_times[engine]
                prep_end = solution.prep_end_times[engine]

                if prep_start <= t < prep_end:
                    factory = solution.engine_to_factory[engine]
                    engines_in_prep[factory.id].append(engine.id)

        # Engines in transit
        engines_in_transit = []
        for engine in problem.engines_list:
            if engine in solution.prep_end_times:
                prep_end = solution.prep_end_times[engine]
                arrival = solution.fal_arrival_times[engine]

                if prep_end <= t < arrival:
                    factory = solution.engine_to_factory[engine]
                    msn = solution.engine_to_msn[engine]
                    fal_id = msn.assembly_line_id
                    engines_in_transit.append((engine.id, factory.id, fal_id))

        # Engines arrived
        engines_arrived = defaultdict(list)
        for engine in problem.engines_list:
            if engine in solution.fal_arrival_times:
                arrival = solution.fal_arrival_times[engine]

                if arrival <= t:
                    msn = solution.engine_to_msn[engine]
                    fal_id = msn.assembly_line_id
                    delay = max(0, arrival - msn.engine_installation_date)
                    engines_arrived[fal_id].append((engine.id, arrival, delay))

        # Cumulative KPIs up to time t
        cumulative_delays = 0.0
        cumulative_cost = 0.0

        for engine in problem.engines_list:
            if engine in solution.fal_arrival_times:
                arrival = solution.fal_arrival_times[engine]

                if arrival <= t:
                    msn = solution.engine_to_msn[engine]
                    delay = max(0, arrival - msn.engine_installation_date)
                    cumulative_delays += delay

                    # Add travel cost
                    factory = solution.engine_to_factory[engine]
                    fal = problem.assembly_lines[msn.assembly_line_id]
                    cost = problem.get_travel_cost(factory, fal)
                    cumulative_cost += cost

        timeline[t] = {
            'engines_in_prep': dict(engines_in_prep),
            'engines_in_transit': engines_in_transit,
            'engines_arrived': dict(engines_arrived),
            'cumulative_delays': cumulative_delays,
            'cumulative_cost': cumulative_cost,
        }

    return timeline


def create_dashboard(
    problem: EnginePreparationProblem,
    solution: EnginePreparationSolution,
    title: str = "Engine Preparation & Allocation - Time-Based Dashboard"
) -> go.Figure:
    """
    Create interactive dashboard with time slider

    Shows:
    - Factory status (engines in prep)
    - Transit status (engines moving)
    - Arrival board (engines delivered)
    - Cumulative KPIs
    """
    timeline = compute_timeline_data(problem, solution)

    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Factory Workload",
            "Engines in Transit",
            "Arrivals at FALs",
            "Cumulative Delays",
            "Factory Details",
            "Cumulative Travel Cost",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "table"}, {"type": "scatter"}],
            [{"type": "scatter", "colspan": 2}, None],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )

    # Prepare frames for animation
    frames = []
    times = sorted(timeline.keys())

    for t in times:
        snapshot = timeline[t]

        # Factory workload (bar chart)
        factory_names = [f.id for f in sorted(problem.factories_list, key=lambda x: x.id)]
        factory_counts = [len(snapshot['engines_in_prep'].get(f, [])) for f in factory_names]
        factory_capacities = [
            problem.factories[f].capacity if problem.factories[f].capacity else 10
            for f in factory_names
        ]

        # Transit count by destination FAL
        transit_by_fal = defaultdict(int)
        for _, _, fal_id in snapshot['engines_in_transit']:
            transit_by_fal[fal_id] += 1

        fal_names = [f.id for f in sorted(problem.assembly_lines_list, key=lambda x: x.id)]
        transit_counts = [transit_by_fal.get(f, 0) for f in fal_names]

        # Arrivals by FAL
        arrival_counts = {fal: len(engines) for fal, engines in snapshot['engines_arrived'].items()}
        arrival_count_list = [arrival_counts.get(f, 0) for f in fal_names]

        # Create frame data
        frame_data = [
            # Row 1, Col 1: Factory workload
            go.Bar(
                x=factory_names,
                y=factory_counts,
                name="Engines in Prep",
                marker=dict(color='steelblue'),
                text=factory_counts,
                textposition='auto',
                hovertemplate="Factory: %{x}<br>Engines: %{y}<extra></extra>",
            ),
            # Capacity line
            go.Scatter(
                x=factory_names,
                y=factory_capacities,
                mode='markers',
                name="Capacity",
                marker=dict(color='red', symbol='line-ew', size=12, line=dict(width=2)),
                hovertemplate="Capacity: %{y}<extra></extra>",
            ),

            # Row 1, Col 2: Transit
            go.Bar(
                x=fal_names,
                y=transit_counts,
                name="In Transit",
                marker=dict(color='orange'),
                text=transit_counts,
                textposition='auto',
                hovertemplate="Destination: %{x}<br>Engines: %{y}<extra></extra>",
            ),

            # Row 2, Col 1: Arrivals
            go.Bar(
                x=fal_names,
                y=arrival_count_list,
                name="Arrived",
                marker=dict(color='green'),
                text=arrival_count_list,
                textposition='auto',
                hovertemplate="FAL: %{x}<br>Arrived: %{y}<extra></extra>",
            ),

            # Row 2, Col 2: Cumulative delays line
            go.Scatter(
                x=times[:times.index(t)+1],
                y=[timeline[tt]['cumulative_delays'] for tt in times[:times.index(t)+1]],
                mode='lines+markers',
                name="Cum. Delays",
                line=dict(color='red', width=3),
                fill='tozeroy',
                hovertemplate="Time: %{x}<br>Delays: %{y:.1f} days<extra></extra>",
            ),

            # Row 3, Col 2: Cumulative cost line
            go.Scatter(
                x=times[:times.index(t)+1],
                y=[timeline[tt]['cumulative_cost'] for tt in times[:times.index(t)+1]],
                mode='lines+markers',
                name="Cum. Cost",
                line=dict(color='green', width=3),
                fill='tozeroy',
                hovertemplate="Time: %{x}<br>Cost: €%{y:.0f}<extra></extra>",
            ),
        ]

        # Build factory details table
        table_headers = ["Factory", "Engines", "Capacity", "Status"]
        table_cells = [[], [], [], []]

        for factory in sorted(problem.factories_list, key=lambda x: x.id):
            engines_here = snapshot['engines_in_prep'].get(factory.id, [])
            count = len(engines_here)
            capacity = factory.capacity if factory.capacity else "∞"

            status = "✓ OK" if (factory.capacity is None or count <= factory.capacity) else "⚠ OVER"

            table_cells[0].append(factory.id)
            table_cells[1].append(", ".join(engines_here) if engines_here else "-")
            table_cells[2].append(str(capacity))
            table_cells[3].append(status)

        table_trace = go.Table(
            header=dict(values=table_headers, fill_color='lightgrey', font=dict(size=12, color='black')),
            cells=dict(values=table_cells, fill_color='white', font=dict(size=10)),
        )

        frame_data.insert(4, table_trace)  # Insert table at position 4

        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Set initial frame data
    fig.add_traces(frames[0].data)

    # Update subplot properties
    fig.update_xaxes(title_text="Factory", row=1, col=1)
    fig.update_yaxes(title_text="Engines", row=1, col=1)

    fig.update_xaxes(title_text="Destination FAL", row=1, col=2)
    fig.update_yaxes(title_text="Engines", row=1, col=2)

    fig.update_xaxes(title_text="FAL", row=2, col=1)
    fig.update_yaxes(title_text="Engines", row=2, col=1)

    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Days", row=2, col=2)

    fig.update_xaxes(title_text="Time (days)", row=3, col=2)
    fig.update_yaxes(title_text="Cost (€)", row=3, col=2)

    # Create slider
    sliders = [dict(
        active=0,
        yanchor="top",
        y=0,
        xanchor="left",
        currentvalue=dict(
            prefix="Time: ",
            visible=True,
            xanchor="right"
        ),
        steps=[dict(
            args=[[f.name], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
            )],
            label=f"{t}",
            method="animate"
        ) for t, f in zip(times, frames)]
    )]

    # Add play/pause buttons
    updatemenus = [dict(
        type="buttons",
        direction="left",
        buttons=[
            dict(
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)],
                label="▶ Play",
                method="animate"
            ),
            dict(
                args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                label="⏸ Pause",
                method="animate"
            )
        ],
        x=0.1,
        xanchor="left",
        y=0,
        yanchor="top"
    )]

    fig.frames = frames

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        sliders=sliders,
        updatemenus=updatemenus,
        height=1200,
        showlegend=False,
        hovermode='closest',
    )

    return fig


def visualize_solution(
    problem: EnginePreparationProblem,
    solution: EnginePreparationSolution,
    save_html: str = None,
) -> None:
    """
    Create and display interactive dashboard

    Args:
        problem: Problem instance
        solution: Solution to visualize
        save_html: Optional path to save HTML file
    """
    print("Creating interactive dashboard...")

    fig = create_dashboard(problem, solution)

    # Show figure
    fig.show()

    # Save if requested
    if save_html:
        fig.write_html(save_html)
        print(f"\n✓ Dashboard saved to: {save_html}")
        print(f"   Open in browser and use the time slider to explore the solution!")
