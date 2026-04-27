"""
Improved Time-Based Dashboard for Engine Preparation and Allocation

Simple, clear dashboard with large, readable charts
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
from collections import defaultdict

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
)


def compute_timeline_data(problem, solution):
    """Compute time-series data for all time points"""
    timeline = {}

    for t in range(0, problem.time_horizon + 1):
        # Engines in preparation
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
                    engines_in_transit.append((engine.id, factory.id, msn.assembly_line_id))

        # Engines arrived
        engines_arrived = defaultdict(list)
        for engine in problem.engines_list:
            if engine in solution.fal_arrival_times:
                arrival = solution.fal_arrival_times[engine]
                if arrival <= t:
                    msn = solution.engine_to_msn[engine]
                    delay = max(0, arrival - msn.engine_installation_date)
                    engines_arrived[msn.assembly_line_id].append((engine.id, arrival, delay))

        # Cumulative KPIs
        cumulative_delays = sum(
            max(0, solution.fal_arrival_times[eng] - solution.engine_to_msn[eng].engine_installation_date)
            for eng in problem.engines_list
            if eng in solution.fal_arrival_times and solution.fal_arrival_times[eng] <= t
        )

        cumulative_cost = sum(
            problem.get_travel_cost(solution.engine_to_factory[eng], problem.assembly_lines[solution.engine_to_msn[eng].assembly_line_id])
            for eng in problem.engines_list
            if eng in solution.fal_arrival_times and solution.fal_arrival_times[eng] <= t
        )

        timeline[t] = {
            'engines_in_prep': dict(engines_in_prep),
            'engines_in_transit': engines_in_transit,
            'engines_arrived': dict(engines_arrived),
            'cumulative_delays': cumulative_delays,
            'cumulative_cost': cumulative_cost,
        }

    return timeline


def create_dashboard(problem, solution):
    """Create 2x2 dashboard with large, clear charts"""

    timeline = compute_timeline_data(problem, solution)
    times = sorted(timeline.keys())

    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Factory Workload (Current)</b>",
            "<b>Cumulative Delays Over Time</b>",
            "<b>Engines in Transit (Current)</b>",
            "<b>Cumulative Travel Cost Over Time</b>",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Prepare animation frames
    frames = []

    for t in times:
        snapshot = timeline[t]

        # Chart 1: Factory Workload (Bar)
        factory_names = [f.id for f in sorted(problem.factories_list, key=lambda x: x.id)]
        factory_counts = [len(snapshot['engines_in_prep'].get(f, [])) for f in factory_names]
        factory_capacities = [
            f.capacity if f.capacity else 0
            for f in sorted(problem.factories_list, key=lambda x: x.id)
        ]

        workload_bar = go.Bar(
            x=factory_names,
            y=factory_counts,
            marker=dict(color='#3498db', line=dict(color='white', width=2)),
            text=factory_counts,
            textposition='outside',
            textfont=dict(size=14, color='black'),
            hovertemplate="<b>%{x}</b><br>Engines in prep: %{y}<extra></extra>",
        )

        # Capacity markers
        capacity_markers = go.Scatter(
            x=factory_names,
            y=factory_capacities,
            mode='markers',
            marker=dict(
                symbol='line-ew',
                size=16,
                color='red',
                line=dict(width=3)
            ),
            name='Capacity',
            hovertemplate="<b>%{x}</b><br>Capacity: %{y}/week<extra></extra>",
        )

        # Chart 2: Cumulative Delays (Line)
        cum_delays_line = go.Scatter(
            x=times[:times.index(t)+1],
            y=[timeline[tt]['cumulative_delays'] for tt in times[:times.index(t)+1]],
            mode='lines+markers',
            line=dict(color='#e74c3c', width=4),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)',
            hovertemplate="Time %{x}: %{y:.1f} days delay<extra></extra>",
        )

        # Chart 3: Transit (Bar)
        transit_by_fal = defaultdict(int)
        for _, _, fal_id in snapshot['engines_in_transit']:
            transit_by_fal[fal_id] += 1

        fal_names = [f.id for f in sorted(problem.assembly_lines_list, key=lambda x: x.id)]
        transit_counts = [transit_by_fal.get(f, 0) for f in fal_names]

        transit_bar = go.Bar(
            x=fal_names,
            y=transit_counts,
            marker=dict(color='#f39c12', line=dict(color='white', width=2)),
            text=transit_counts,
            textposition='outside',
            textfont=dict(size=14, color='black'),
            hovertemplate="<b>%{x}</b><br>Engines in transit: %{y}<extra></extra>",
        )

        # Chart 4: Cumulative Cost (Line)
        cum_cost_line = go.Scatter(
            x=times[:times.index(t)+1],
            y=[timeline[tt]['cumulative_cost'] for tt in times[:times.index(t)+1]],
            mode='lines+markers',
            line=dict(color='#27ae60', width=4),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.2)',
            hovertemplate="Time %{x}: €%{y:,.0f}<extra></extra>",
        )

        # Create frame - must specify row/col for each trace
        # Both workload_bar and capacity_markers go to row=1, col=1
        # Need to create list with all traces in the order they'll be added
        frame_data = [
            workload_bar,      # Row 1, Col 1
            capacity_markers,  # Row 1, Col 1 (same subplot as workload_bar)
            cum_delays_line,   # Row 1, Col 2
            transit_bar,       # Row 2, Col 1
            cum_cost_line,     # Row 2, Col 2
        ]

        frames.append(go.Frame(data=frame_data, name=str(t)))

    # Add initial frame data with explicit row/col assignment
    initial_data = frames[0].data
    fig.add_trace(initial_data[0], row=1, col=1)  # workload_bar
    fig.add_trace(initial_data[1], row=1, col=1)  # capacity_markers (same subplot)
    fig.add_trace(initial_data[2], row=1, col=2)  # cum_delays_line
    fig.add_trace(initial_data[3], row=2, col=1)  # transit_bar
    fig.add_trace(initial_data[4], row=2, col=2)  # cum_cost_line

    # Update axes with larger fonts
    fig.update_xaxes(title_text="<b>Factory</b>", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=1, tickangle=-45)
    fig.update_yaxes(title_text="<b>Engines</b>", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=1, range=[0, max(20, problem.time_horizon//5)])

    fig.update_xaxes(title_text="<b>Time (days)</b>", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=2)
    fig.update_yaxes(title_text="<b>Total Delay (days)</b>", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=2)

    fig.update_xaxes(title_text="<b>Destination FAL</b>", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=1, tickangle=-45)
    fig.update_yaxes(title_text="<b>Engines</b>", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=1, range=[0, max(20, len(problem.engines_list)//2)])

    fig.update_xaxes(title_text="<b>Time (days)</b>", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=2)
    fig.update_yaxes(title_text="<b>Total Cost (€)</b>", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=2)

    # Add annotations showing current time and KPIs
    annotations_template = [
        dict(
            text=f"<b>Time: {{time}} days</b><br>Active Factories: {{active_factories}}<br>In Transit: {{in_transit}}<br>Arrived: {{arrived}}",
            xref="paper", yref="paper",
            x=0.5, y=1.08,
            xanchor='center', yanchor='bottom',
            font=dict(size=18, color='#2c3e50'),
            showarrow=False,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#2c3e50',
            borderwidth=2,
            borderpad=10,
        )
    ]

    # Create slider
    sliders = [dict(
        active=0,
        yanchor="top",
        y=-0.05,
        xanchor="left",
        x=0.15,
        currentvalue=dict(
            prefix="<b>Time: </b>",
            visible=True,
            xanchor="center",
            font=dict(size=18)
        ),
        pad=dict(b=10, t=50),
        len=0.7,
        steps=[dict(
            args=[[f.name], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
                transition=dict(duration=0)
            )],
            label=f"<b>{t}</b>" if t % 10 == 0 else str(t),
            method="animate"
        ) for t, f in zip(times, frames)]
    )]

    # Add play/pause buttons
    updatemenus = [dict(
        type="buttons",
        direction="left",
        buttons=[
            dict(
                args=[None, dict(
                    frame=dict(duration=300, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0)
                )],
                label="▶ PLAY",
                method="animate"
            ),
            dict(
                args=[[None], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0)
                )],
                label="⏸ PAUSE",
                method="animate"
            )
        ],
        x=0.05,
        xanchor="left",
        y=-0.05,
        yanchor="top",
        font=dict(size=14),
        bgcolor='#ecf0f1',
        bordercolor='#2c3e50',
        borderwidth=2,
    )]

    fig.frames = frames

    fig.update_layout(
        title=dict(
            text="<b>Engine Preparation & Allocation Dashboard</b><br><sub>Use slider to navigate through time</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2c3e50')
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        height=900,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#ecf0f1',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(t=120, b=100, l=80, r=80),
    )

    return fig


def visualize_solution(problem, solution, save_html=None):
    """Create and display dashboard"""
    print("Creating interactive dashboard...")

    fig = create_dashboard(problem, solution)
    fig.show()

    if save_html:
        fig.write_html(save_html)
        print(f"\n✓ Dashboard saved to: {save_html}")
        print(f"   📊 Features:")
        print(f"      - Time slider to navigate through solution")
        print(f"      - Factory workload shows capacity constraints")
        print(f"      - Cumulative KPIs track performance over time")
        print(f"   🎬 Click ▶ PLAY to animate!")
