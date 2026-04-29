"""
Enhanced Comparison Dashboard: JIT Theoretical vs CP-SAT Optimized Solution

Compares the theoretical JIT production load with the optimized solution.
Shows objective trade-offs and factory preparation switches.

Run with: streamlit run part_preparation/comparison_dashboard.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import defaultdict
import numpy as np
from typing import Dict

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
)


def compute_jit_theoretical_load(problem):
    """
    Compute theoretical JIT production load.

    For each MSN, computes when prep must happen at the preferred factory
    to have the engine arrive EXACTLY at the installation date (pure JIT).

    This shows the theoretical minimum capacity required and highlights bottlenecks.
    """
    factory_load = defaultdict(lambda: defaultdict(int))

    for msn in problem.msns_list:
        # Find preferred factory for this MSN's engine type and FAL
        compatible_factories = [
            f for f in problem.factories_list
            if msn.engine_type in f.compatible_engine_types
        ]

        # Prefer factory assigned to this MSN's FAL
        preferred_factories = [
            f for f in compatible_factories
            if f.default_assembly_line.id == msn.assembly_line_id
        ]

        if not preferred_factories:
            continue

        factory = preferred_factories[0]

        # Get preparation duration from factory data
        prep_duration = factory.get_preparation_duration(msn.engine_type)

        # Get travel time from factory to FAL
        assembly_line = problem.assembly_lines[msn.assembly_line_id]
        travel_time = problem.get_travel_time(factory, assembly_line)

        # Work backwards from installation date (pure JIT - arrive exactly when needed)
        arrival_time = msn.engine_installation_date  # Arrive exactly at installation time
        prep_end = arrival_time - travel_time        # Must finish prep to allow travel
        prep_start = prep_end - prep_duration         # Start prep this many days before

        # Record factory load during preparation period
        for t in range(max(0, prep_start), min(problem.time_horizon, prep_end)):
            factory_load[factory.id][t] += 1

    return factory_load


def compute_optimized_load(problem, solution):
    """Compute actual production load from CP-SAT solution"""
    factory_load = defaultdict(lambda: defaultdict(int))

    for engine in problem.engines_list:
        if engine not in solution.engine_to_factory:
            continue  # Unallocated engine

        factory = solution.engine_to_factory[engine]
        prep_start = solution.prep_start_times[engine]
        prep_end = solution.prep_end_times[engine]

        for t in range(prep_start, prep_end):
            factory_load[factory.id][t] += 1

    return factory_load


def count_factory_switches(problem, solution):
    """
    Count the number of times engines switch factories compared to their preferred factory
    Also computes transport costs by factory
    """
    switches = 0
    transport_costs_by_factory = defaultdict(float)
    switch_details = []

    for engine in problem.engines_list:
        if engine not in solution.engine_to_msn:
            continue  # Unallocated

        msn = solution.engine_to_msn[engine]
        actual_factory = solution.engine_to_factory[engine]

        # Find preferred factory
        compatible_factories = [
            f for f in problem.factories_list
            if msn.engine_type in f.compatible_engine_types
        ]

        preferred_factories = [
            f for f in compatible_factories
            if f.default_assembly_line.id == msn.assembly_line_id
        ]

        if preferred_factories:
            preferred_factory = preferred_factories[0]
            if actual_factory.id != preferred_factory.id:
                switches += 1
                switch_details.append({
                    'engine': engine.id,
                    'msn': msn.id,
                    'preferred': preferred_factory.id,
                    'actual': actual_factory.id,
                    'fal': msn.assembly_line_id,
                })

        # Compute transport cost
        assembly_line = problem.assembly_lines[msn.assembly_line_id]
        transport_cost = problem.get_travel_cost(actual_factory, assembly_line)
        transport_costs_by_factory[actual_factory.id] += transport_cost

    return switches, switch_details, transport_costs_by_factory


def create_comparison_chart(problem, jit_load, optimized_load):
    """
    Create comparison chart: JIT theoretical load vs Optimized load
    """
    factories = sorted(problem.factories_list, key=lambda f: f.id)
    n_factories = len(factories)

    fig = make_subplots(
        rows=n_factories,
        cols=1,
        subplot_titles=[
            f"<b>{f.id}</b> (Capacity={f.default_capacity}, FAL={f.default_assembly_line.id})"
            for f in factories
        ],
        vertical_spacing=0.05,
        shared_xaxes=True,
    )

    for idx, factory in enumerate(factories):
        row = idx + 1

        # Get JIT load timeline
        jit_times = sorted(jit_load.get(factory.id, {}).keys())
        jit_loads = [jit_load[factory.id][t] for t in jit_times] if jit_times else []

        # Get optimized load timeline
        opt_times = sorted(optimized_load.get(factory.id, {}).keys())
        opt_loads = [optimized_load[factory.id][t] for t in opt_times] if opt_times else []

        # JIT load (theoretical)
        if jit_times:
            fig.add_trace(
                go.Scatter(
                    x=jit_times,
                    y=jit_loads,
                    mode='lines',
                    name='JIT Theoretical',
                    line=dict(color='#e74c3c', width=2, dash='dot'),
                    fill='tozeroy',
                    fillcolor='rgba(231, 76, 60, 0.2)',
                    showlegend=(idx == 0),
                ),
                row=row, col=1
            )

        # Optimized load (CP-SAT solution)
        if opt_times:
            fig.add_trace(
                go.Scatter(
                    x=opt_times,
                    y=opt_loads,
                    mode='lines',
                    name='CP-SAT Optimized',
                    line=dict(color='#3498db', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.3)',
                    showlegend=(idx == 0),
                ),
                row=row, col=1
            )

        # Capacity line
        if factory.default_capacity is not None:
            fig.add_trace(
                go.Scatter(
                    x=[0, problem.time_horizon],
                    y=[factory.default_capacity, factory.default_capacity],
                    mode='lines',
                    name='Capacity',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=(idx == 0),
                ),
                row=row, col=1
            )

        # Update y-axis
        fig.update_yaxes(
            title_text="Engines",
            row=row, col=1,
            title_font=dict(size=12),
            tickfont=dict(size=10)
        )

    # Update x-axis (only last subplot)
    fig.update_xaxes(
        title_text="<b>Time (days)</b>",
        row=n_factories, col=1,
        title_font=dict(size=14),
        tickfont=dict(size=11)
    )

    fig.update_layout(
        height=300 * n_factories,
        title=dict(
            text="<b>Factory Load Comparison: JIT vs Optimized</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100, b=50, l=80, r=50)
    )

    return fig


def create_objective_tradeoff_plot(problem, solution):
    """
    Create 2D scatter plots showing trade-offs between objectives
    """
    kpis = problem.evaluate(solution)

    # Create data for comparison
    objectives = {
        'Baseline Changes': kpis.get('baseline_changes', 0),
        'Arrival Delays': kpis.get('arrival_delays', 0),
        'Travel Cost': kpis.get('travel_cost', 0),
        'Storage Cost': kpis.get('storage_cost', 0),
        'Unallocated MSNs': kpis.get('unallocated_msns', 0),
    }

    # Create subplots for different objective combinations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Travel Cost vs Delays</b>",
            "<b>Storage Cost vs Delays</b>",
            "<b>Baseline Changes vs Delays</b>",
            "<b>Unallocated MSNs vs Total Cost</b>",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Plot 1: Travel Cost vs Delays
    fig.add_trace(
        go.Scatter(
            x=[objectives['Travel Cost']],
            y=[objectives['Arrival Delays']],
            mode='markers',
            marker=dict(size=20, color='#e74c3c', symbol='diamond'),
            name='Current Solution',
            showlegend=False,
        ),
        row=1, col=1
    )

    # Plot 2: Storage Cost vs Delays
    fig.add_trace(
        go.Scatter(
            x=[objectives['Storage Cost']],
            y=[objectives['Arrival Delays']],
            mode='markers',
            marker=dict(size=20, color='#3498db', symbol='diamond'),
            name='Current Solution',
            showlegend=False,
        ),
        row=1, col=2
    )

    # Plot 3: Baseline Changes vs Delays
    fig.add_trace(
        go.Scatter(
            x=[objectives['Baseline Changes']],
            y=[objectives['Arrival Delays']],
            mode='markers',
            marker=dict(size=20, color='#f39c12', symbol='diamond'),
            name='Current Solution',
            showlegend=False,
        ),
        row=2, col=1
    )

    # Plot 4: Unallocated MSNs vs Total Cost
    total_cost = objectives['Travel Cost'] + objectives['Storage Cost']
    fig.add_trace(
        go.Scatter(
            x=[objectives['Unallocated MSNs']],
            y=[total_cost],
            mode='markers',
            marker=dict(size=20, color='#9b59b6', symbol='diamond'),
            name='Current Solution',
            showlegend=False,
        ),
        row=2, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="<b>Travel Cost (€)</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Arrival Delays (days)</b>", row=1, col=1)

    fig.update_xaxes(title_text="<b>Storage Cost (€)</b>", row=1, col=2)
    fig.update_yaxes(title_text="<b>Arrival Delays (days)</b>", row=1, col=2)

    fig.update_xaxes(title_text="<b>Baseline Changes</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Arrival Delays (days)</b>", row=2, col=1)

    fig.update_xaxes(title_text="<b>Unallocated MSNs</b>", row=2, col=2)
    fig.update_yaxes(title_text="<b>Total Cost (€)</b>", row=2, col=2)

    fig.update_layout(
        height=800,
        title=dict(
            text="<b>Objective Trade-offs Analysis</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#ecf0f1',
        paper_bgcolor='white',
    )

    return fig


def main():
    st.set_page_config(
        page_title="JIT vs Optimized Comparison",
        page_icon="🔄",
        layout="wide"
    )

    st.title("🔄 Engine Preparation: JIT vs Optimized Comparison Dashboard")

    # Check if data is loaded
    if 'problem' not in st.session_state or 'solution' not in st.session_state:
        st.warning("⚠️ No solution loaded. Please load problem and solution data.")
        st.info("""
        To use this dashboard:
        1. Run your CP-SAT solver to generate a solution
        2. Save the problem and solution to session state:
        ```python
        import streamlit as st
        st.session_state.problem = problem
        st.session_state.solution = cpsat_solution
        ```
        3. Then run: `streamlit run part_preparation/comparison_dashboard.py`
        """)
        return

    problem = st.session_state.problem
    solution = st.session_state.solution

    # Sidebar controls
    st.sidebar.header("⚙️ Analysis Parameters")

    st.sidebar.info("""
    **JIT Theoretical Load**:
    Shows the production load if engines were prepared Just-In-Time
    to arrive EXACTLY at installation date, using actual factory
    preparation durations and travel times from problem data.

    This represents the theoretical minimum capacity needed.
    """)

    # Compute loads (JIT with no configurable lead time - pure JIT)
    jit_load = compute_jit_theoretical_load(problem)
    optimized_load = compute_optimized_load(problem, solution)

    # Compute factory switches
    switches, switch_details, transport_costs = count_factory_switches(problem, solution)

    # Evaluate solution
    kpis = problem.evaluate(solution)

    # Summary metrics
    st.header("📊 Solution Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Baseline Changes", f"{kpis.get('baseline_changes', 0):.0f}")
        st.caption("Changes from baseline")

    with col2:
        st.metric("Arrival Delays", f"{kpis.get('arrival_delays', 0):.1f} days")
        st.caption("Total delays to MSNs")

    with col3:
        st.metric("Travel Cost", f"€{kpis.get('travel_cost', 0):,.2f}")
        st.caption("Factory → FAL transport")

    with col4:
        st.metric("Storage Cost", f"€{kpis.get('storage_cost', 0):,.2f}")
        st.caption("Engine storage time")

    with col5:
        st.metric("Unallocated MSNs", f"{kpis.get('unallocated_msns', 0):.0f}")
        st.caption("MSNs without engines")

    # Factory switches
    st.header("🔄 Factory Preparation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Factory Switches", f"{switches}")
        st.caption("Engines sent to non-preferred factories")

    with col2:
        feasible = problem.satisfy(solution)
        st.metric("Feasible", "✅ Yes" if feasible else "❌ No")
        st.caption("All constraints satisfied")

    # Show switch details
    if switch_details:
        with st.expander(f"📋 Factory Switch Details ({len(switch_details)} switches)"):
            switch_df = pd.DataFrame(switch_details)
            st.dataframe(switch_df, use_container_width=True)

    # Transport costs by factory
    st.subheader("💰 Transport Costs by Factory")

    if transport_costs:
        cost_df = pd.DataFrame([
            {'Factory': factory_id, 'Transport Cost (€)': f"{cost:,.2f}"}
            for factory_id, cost in sorted(transport_costs.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

    # Comparison chart
    st.header("📈 Factory Load Comparison: JIT vs Optimized")

    fig = create_comparison_chart(problem, jit_load, optimized_load)
    st.plotly_chart(fig, use_container_width=True)

    # Objective trade-offs
    st.header("🎯 Objective Trade-offs Analysis")

    tradeoff_fig = create_objective_tradeoff_plot(problem, solution)
    st.plotly_chart(tradeoff_fig, use_container_width=True)

    # Capacity utilization comparison
    st.header("📊 Capacity Utilization Comparison")

    utilization_data = []
    for factory in problem.factories_list:
        jit_load_factory = jit_load.get(factory.id, {})
        opt_load_factory = optimized_load.get(factory.id, {})

        if factory.default_capacity is None:
            continue

        # JIT utilization
        if jit_load_factory:
            jit_avg = np.mean(list(jit_load_factory.values()))
            jit_max = max(jit_load_factory.values())
            jit_peak_util = jit_max / factory.default_capacity * 100
        else:
            jit_avg = 0
            jit_max = 0
            jit_peak_util = 0

        # Optimized utilization
        if opt_load_factory:
            opt_avg = np.mean(list(opt_load_factory.values()))
            opt_max = max(opt_load_factory.values())
            opt_peak_util = opt_max / factory.default_capacity * 100
        else:
            opt_avg = 0
            opt_max = 0
            opt_peak_util = 0

        utilization_data.append({
            'Factory': factory.id,
            'FAL': factory.default_assembly_line.id,
            'Capacity': factory.default_capacity,
            'JIT Peak Util (%)': f"{jit_peak_util:.1f}",
            'Optimized Peak Util (%)': f"{opt_peak_util:.1f}",
            'JIT Max Load': jit_max,
            'Optimized Max Load': opt_max,
            'Improvement': '✅' if opt_peak_util < jit_peak_util else ('⚠️' if opt_peak_util == jit_peak_util else '❌'),
        })

    if utilization_data:
        util_df = pd.DataFrame(utilization_data)
        st.dataframe(
            util_df.sort_values('Optimized Peak Util (%)', ascending=False),
            use_container_width=True,
            hide_index=True
        )

    # Additional metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Problem Size:**")
    st.sidebar.write(f"- {len(problem.assembly_lines_list)} assembly lines")
    st.sidebar.write(f"- {len(problem.factories_list)} factories")
    st.sidebar.write(f"- {len(problem.engines_list)} engines")
    st.sidebar.write(f"- {len(problem.msns_list)} MSNs")


if __name__ == "__main__":
    main()