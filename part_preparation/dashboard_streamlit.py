"""
Streamlit Dashboard for Engine Preparation and Allocation

Run with: streamlit run part_preparation/dashboard_streamlit.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from typing import Dict, List

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


def create_gantt_chart(problem, solution, current_time):
    """Create Gantt chart showing engine preparation timeline"""

    fig = go.Figure()

    # Organize factories by location
    factories_sorted = sorted(problem.factories_list, key=lambda f: (f.location, f.id))

    # Color map for engine types
    engine_type_colors = {
        "EngineType_1": "#3498db",
        "EngineType_2": "#e74c3c",
        "TrentXWB": "#3498db",
        "GE90": "#e74c3c",
    }

    for factory_idx, factory in enumerate(factories_sorted):
        # Find all engines prepared at this factory
        engines_at_factory = [
            (engine, solution.prep_start_times[engine], solution.prep_end_times[engine])
            for engine in problem.engines_list
            if engine in solution.engine_to_factory and solution.engine_to_factory[engine] == factory
        ]

        # Sort by start time
        engines_at_factory.sort(key=lambda x: x[1])

        for engine, start, end in engines_at_factory:
            # Determine if engine is currently active
            is_active = start <= current_time < end

            # Get engine color
            color = engine_type_colors.get(engine.engine_type, "#95a5a6")
            if is_active:
                # Highlight active engines with brighter color and border
                opacity = 1.0
                line_width = 3
                line_color = "yellow"
            else:
                opacity = 0.6
                line_width = 1
                line_color = "white"

            fig.add_trace(go.Bar(
                x=[end - start],
                y=[factory.id],
                base=[start],
                orientation='h',
                marker=dict(
                    color=color,
                    opacity=opacity,
                    line=dict(color=line_color, width=line_width)
                ),
                text=engine.id,
                textposition='inside',
                textfont=dict(size=10, color='white'),
                hovertemplate=(
                    f"<b>{engine.id}</b><br>"
                    f"Type: {engine.engine_type}<br>"
                    f"Factory: {factory.id}<br>"
                    f"Start: {start}<br>"
                    f"End: {end}<br>"
                    f"Duration: {end - start} days<br>"
                    f"Status: {'🟢 ACTIVE' if is_active else '⚪ Completed' if end <= current_time else '⏳ Scheduled'}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

    # Add vertical line for current time
    fig.add_vline(
        x=current_time,
        line=dict(color="red", width=3, dash="dash"),
        annotation_text=f"Now (t={current_time})",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"<b>Engine Preparation Gantt Chart</b>",
        xaxis_title="<b>Time (days)</b>",
        yaxis_title="<b>Factory</b>",
        height=max(400, len(factories_sorted) * 40),
        barmode='overlay',
        hovermode='closest',
        plot_bgcolor='#ecf0f1',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        xaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=13),
            range=[0, problem.time_horizon]
        ),
        yaxis=dict(
            title_font=dict(size=16),
            tickfont=dict(size=13),
            autorange="reversed"
        ),
    )

    return fig


def create_dashboard(problem, solution, current_time):
    """Create dashboard for a specific time point"""

    timeline = compute_timeline_data(problem, solution)
    times = sorted(timeline.keys())

    # Get current snapshot
    snapshot = timeline[current_time]

    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"<b>Factory Workload (t={current_time})</b>",
            "<b>Cumulative Delays Over Time</b>",
            f"<b>Engines in Transit (t={current_time})</b>",
            "<b>Cumulative Travel Cost Over Time</b>",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Chart 1: Factory Workload (Bar)
    factory_names = [f.id for f in sorted(problem.factories_list, key=lambda x: x.id)]
    factory_counts = [len(snapshot['engines_in_prep'].get(f, [])) for f in factory_names]
    factory_capacities = [
        f.capacity if f.capacity else 0
        for f in sorted(problem.factories_list, key=lambda x: x.id)
    ]

    fig.add_trace(
        go.Bar(
            x=factory_names,
            y=factory_counts,
            marker=dict(color='#3498db', line=dict(color='white', width=2)),
            text=factory_counts,
            textposition='outside',
            textfont=dict(size=14, color='black'),
            hovertemplate="<b>%{x}</b><br>Engines in prep: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1
    )

    # Capacity markers
    fig.add_trace(
        go.Scatter(
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
            showlegend=False,
        ),
        row=1, col=1
    )

    # Chart 2: Cumulative Delays (Line)
    times_up_to_now = [t for t in times if t <= current_time]
    delays_up_to_now = [timeline[t]['cumulative_delays'] for t in times_up_to_now]

    fig.add_trace(
        go.Scatter(
            x=times_up_to_now,
            y=delays_up_to_now,
            mode='lines+markers',
            line=dict(color='#e74c3c', width=4),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)',
            hovertemplate="Time %{x}: %{y:.1f} days delay<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2
    )

    # Chart 3: Transit (Bar)
    transit_by_fal = defaultdict(int)
    for _, _, fal_id in snapshot['engines_in_transit']:
        transit_by_fal[fal_id] += 1

    fal_names = [f.id for f in sorted(problem.assembly_lines_list, key=lambda x: x.id)]
    transit_counts = [transit_by_fal.get(f, 0) for f in fal_names]

    fig.add_trace(
        go.Bar(
            x=fal_names,
            y=transit_counts,
            marker=dict(color='#f39c12', line=dict(color='white', width=2)),
            text=transit_counts,
            textposition='outside',
            textfont=dict(size=14, color='black'),
            hovertemplate="<b>%{x}</b><br>Engines in transit: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1
    )

    # Chart 4: Cumulative Cost (Line)
    costs_up_to_now = [timeline[t]['cumulative_cost'] for t in times_up_to_now]

    fig.add_trace(
        go.Scatter(
            x=times_up_to_now,
            y=costs_up_to_now,
            mode='lines+markers',
            line=dict(color='#27ae60', width=4),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(39, 174, 96, 0.2)',
            hovertemplate="Time %{x}: €%{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="<b>Factory</b>", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=1, tickangle=-45)
    fig.update_yaxes(title_text="<b>Engines</b>", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=1, range=[0, max(10, max(factory_counts) + 2) if factory_counts else 10])

    fig.update_xaxes(title_text="<b>Time (days)</b>", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=2)
    fig.update_yaxes(title_text="<b>Total Delay (days)</b>", title_font=dict(size=16), tickfont=dict(size=13), row=1, col=2)

    fig.update_xaxes(title_text="<b>Destination FAL</b>", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=1, tickangle=-45)
    fig.update_yaxes(title_text="<b>Engines</b>", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=1, range=[0, max(10, max(transit_counts) + 2) if transit_counts else 10])

    fig.update_xaxes(title_text="<b>Time (days)</b>", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=2)
    fig.update_yaxes(title_text="<b>Total Cost (€)</b>", title_font=dict(size=16), tickfont=dict(size=13), row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>Engine Preparation & Allocation Dashboard</b><br><sub>Time: {current_time} / {problem.time_horizon} days</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2c3e50')
        ),
        height=900,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#ecf0f1',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(t=120, b=100, l=80, r=80),
    )

    return fig


def main():
    st.set_page_config(
        page_title="Engine Preparation Dashboard",
        page_icon="✈️",
        layout="wide"
    )

    st.title("✈️ Engine Preparation & Allocation Dashboard")

    # Check if data is loaded
    if 'problem' not in st.session_state or 'solution' not in st.session_state:
        st.warning("⚠️ No solution loaded. Please load problem and solution data.")
        st.info("""
        To use this dashboard:
        1. Run your solver to generate a solution
        2. Save the problem and solution to session state
        3. Or modify the example script to launch this dashboard

        Example:
        ```python
        import streamlit as st
        st.session_state.problem = problem
        st.session_state.solution = cpsat_solution
        ```
        """)
        return

    problem = st.session_state.problem
    solution = st.session_state.solution

    # Sidebar with info
    st.sidebar.header("📊 Solution Summary")
    kpis = problem.evaluate(solution)

    st.sidebar.metric("Baseline Changes", f"{kpis['baseline_changes']:.0f}")
    st.sidebar.metric("Total Delays", f"{kpis['arrival_delays']:.1f} days")
    st.sidebar.metric("Travel Cost", f"€{kpis['travel_cost']:.2f}")
    st.sidebar.metric("Feasible", "✅ Yes" if problem.satisfy(solution) else "❌ No")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Problem Size:**")
    st.sidebar.write(f"- {len(problem.assembly_lines_list)} assembly lines")
    st.sidebar.write(f"- {len(problem.factories_list)} factories")
    st.sidebar.write(f"- {len(problem.engines_list)} engines")
    st.sidebar.write(f"- {len(problem.msns_list)} MSNs")

    # Time slider
    st.sidebar.markdown("---")
    current_time = st.sidebar.slider(
        "⏱️ Time (days)",
        min_value=0,
        max_value=problem.time_horizon,
        value=0,
        step=1,
        help="Slide to see the system state at different times"
    )

    # Main dashboard
    fig = create_dashboard(problem, solution, current_time)
    st.plotly_chart(fig, use_container_width=True)

    # Details section
    with st.expander("📋 Details at current time"):
        timeline = compute_timeline_data(problem, solution)
        snapshot = timeline[current_time]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🏭 In Preparation")
            for factory_id, engines in snapshot['engines_in_prep'].items():
                st.write(f"**{factory_id}**: {', '.join(engines) if engines else 'None'}")

        with col2:
            st.subheader("🚚 In Transit")
            if snapshot['engines_in_transit']:
                for eng_id, factory_id, fal_id in snapshot['engines_in_transit']:
                    st.write(f"{eng_id}: {factory_id} → {fal_id}")
            else:
                st.write("None")

        with col3:
            st.subheader("✈️ Arrived")
            for fal_id, arrivals in snapshot['engines_arrived'].items():
                st.write(f"**{fal_id}**: {len(arrivals)} engines")


if __name__ == "__main__":
    main()
