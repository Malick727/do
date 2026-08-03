"""
Unified Dashboard: JIT Analysis + Solution Visualization + Gantt Chart

Run with: streamlit run part_preparation/unified_dashboard.py
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


# ============================================================================
# JIT Analysis Functions (from comparison_dashboard.py)
# ============================================================================

def compute_jit_theoretical_load(problem):
    """Compute theoretical JIT production load"""
    factory_load = defaultdict(lambda: defaultdict(int))

    for msn in problem.msns_list:
        compatible_factories = [
            f for f in problem.factories_list
            if msn.engine_type in f.compatible_engine_types
        ]

        preferred_factories = [
            f for f in compatible_factories
            if f.default_assembly_line.id == msn.assembly_line_id
        ]

        if not preferred_factories:
            continue

        factory = preferred_factories[0]
        prep_duration = factory.get_preparation_duration(msn.engine_type)
        assembly_line = problem.assembly_lines[msn.assembly_line_id]
        travel_time = problem.get_travel_time(factory, assembly_line)

        arrival_time = msn.engine_installation_date
        prep_end = arrival_time - travel_time
        prep_start = prep_end - prep_duration

        for t in range(max(0, prep_start), min(problem.time_horizon, prep_end)):
            factory_load[factory.id][t] += 1

    return factory_load


def compute_optimized_load(problem, solution):
    """
    Calcule la charge (load) de chaque usine au fil du temps pour la solution donnée.
    Gère de façon sécurisée les moteurs non alloués (None).
    """
    # 1. Initialisation de la charge à 0 pour chaque usine et chaque jour
    # On s'assure que time_horizon est bien un entier
    time_horizon = int(problem.time_horizon)
    optimized_load = {f.id: [0] * time_horizon for f in problem.factories_list}

    # 2. Remplissage de la charge
    for engine in problem.engines_list:
        prep_start = solution.prep_start_times.get(engine)
        prep_end = solution.prep_end_times.get(engine)
        factory = solution.engine_to_factory.get(engine)

        # 🔴 SÉCURITÉ 1 : On ignore les moteurs non alloués (ce qui causait ton erreur)
        if prep_start is None or prep_end is None or factory is None:
            continue

        start_t = int(prep_start)
        end_t = int(prep_end)

        # 🔴 SÉCURITÉ 2 : On empêche les erreurs "IndexError" si CP-SAT
        # a planifié une fin au-delà de l'horizon de temps maximum
        start_t = max(0, start_t)
        end_t = min(end_t, time_horizon)

        # 3. Ajout de la charge sur l'intervalle
        for t in range(start_t, end_t):
            optimized_load[factory.id][t] += 1

    return optimized_load

def count_factory_switches(problem, solution):
    """Count engines sent to non-preferred factories"""
    switches = 0
    transport_costs_by_factory = defaultdict(float)
    switch_details = []

    for engine in problem.engines_list:
        if engine not in solution.engine_to_msn:
            continue

        msn = solution.engine_to_msn[engine]
        if msn is None:  # Unallocated engine
            continue

        actual_factory = solution.engine_to_factory[engine]

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

        assembly_line = problem.assembly_lines[msn.assembly_line_id]
        transport_cost = problem.get_travel_cost(actual_factory, assembly_line)
        transport_costs_by_factory[actual_factory.id] += transport_cost

    return switches, switch_details, transport_costs_by_factory


def create_comparison_chart(problem, jit_load, optimized_load):
    """Create comparison chart: JIT theoretical load vs Optimized load"""
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

        jit_times = sorted(jit_load.get(factory.id, {}).keys())
        jit_loads = [jit_load[factory.id][t] for t in jit_times] if jit_times else []

        opt_times = sorted(optimized_load.get(factory.id, {}).keys())
        opt_loads = [optimized_load[factory.id][t] for t in opt_times] if opt_times else []

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

        fig.update_yaxes(
            title_text="Engines",
            row=row, col=1,
            title_font=dict(size=12),
            tickfont=dict(size=10)
        )

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


# ============================================================================
# Gantt Chart Functions (from dashboard_streamlit.py)
# ============================================================================

def create_gantt_chart(problem, solution, current_time):
    """Create Gantt chart showing engine preparation timeline with vertical lanes"""
    fig = go.Figure()

    factories_sorted = sorted(problem.factories_list, key=lambda f: (f.location, f.id))

    engine_type_colors = {
        "EngineType_1": "#3498db",
        "EngineType_2": "#e74c3c",
    }

    y_labels = []
    y_positions = []
    current_y = 0

    for factory_idx, factory in enumerate(factories_sorted):
        engines_at_factory = [
            (engine, solution.prep_start_times[engine], solution.prep_end_times[engine])
            for engine in problem.engines_list
            if engine in solution.engine_to_factory and solution.engine_to_factory[engine] == factory
        ]

        if not engines_at_factory:
            continue

        engines_at_factory.sort(key=lambda x: x[1])

        # Assign lanes to avoid overlaps
        engine_lanes = []
        lanes = []

        for engine, start, end in engines_at_factory:
            assigned_lane = None
            for lane_idx, lane_end_time in enumerate(lanes):
                if start >= lane_end_time:
                    assigned_lane = lane_idx
                    lanes[lane_idx] = end
                    break

            if assigned_lane is None:
                assigned_lane = len(lanes)
                lanes.append(end)

            engine_lanes.append((engine, start, end, assigned_lane))

        num_lanes = len(lanes)
        factory_y_start = current_y
        factory_y_end = current_y + num_lanes - 1

        for engine, start, end, lane in engine_lanes:
            y_pos = current_y + lane
            is_active = start <= current_time < end

            color = engine_type_colors.get(engine.engine_type, "#95a5a6")
            opacity = 1.0 if is_active else 0.6
            line_width = 3 if is_active else 1
            line_color = "yellow" if is_active else "white"

            fig.add_trace(go.Bar(
                x=[end - start],
                y=[y_pos],
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

        factory_y_center = (factory_y_start + factory_y_end) / 2
        y_labels.append(f"{factory.id} ({num_lanes})")
        y_positions.append(factory_y_center)

        if factory_idx < len(factories_sorted) - 1:
            fig.add_hline(
                y=factory_y_end + 0.5,
                line=dict(color="gray", width=1, dash="dot"),
                layer="below"
            )

        current_y += num_lanes

    fig.add_vline(
        x=current_time,
        line=dict(color="red", width=3, dash="dash"),
        annotation_text=f"Now (t={current_time})",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"<b>Engine Preparation Gantt Chart</b>",
        xaxis_title="<b>Time (days)</b>",
        yaxis_title="<b>Factory (lanes)</b>",
        height=max(400, current_y * 30 + 100),
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
            tickmode='array',
            tickvals=y_positions,
            ticktext=y_labels,
            autorange="reversed"
        ),
    )

    return fig


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    st.set_page_config(
        page_title="Engine Preparation Dashboard",
        page_icon="✈️",
        layout="wide"
    )

    st.title("✈️ Engine Preparation & Allocation - Unified Dashboard")

    # Check if data is loaded
    if 'problem' not in st.session_state or 'solution' not in st.session_state:
        st.warning("⚠️ No solution loaded.")
        st.info("""
        Run the solver script to generate and load a solution:
        ```bash
        uv run python examples_do_usecases/part_preparation/run_cpsat_and_visualize.py --instance medium
        ```
        """)
        return

    problem = st.session_state.problem
    solution = st.session_state.solution

    # Sidebar metrics
    st.sidebar.header("📊 Solution Summary")
    kpis = problem.evaluate(solution)

    st.sidebar.metric("Baseline Changes", f"{kpis.get('baseline_changes', 0):.0f}")
    st.sidebar.metric("Arrival Delays", f"{kpis.get('arrival_delays', 0):.1f} days")
    st.sidebar.metric("Travel Cost", f"€{kpis.get('travel_cost', 0):,.2f}")
    st.sidebar.metric("Storage Cost", f"€{kpis.get('storage_cost', 0):,.2f}")
    st.sidebar.metric("Unallocated MSNs", f"{kpis.get('unallocated_msns', 0):.0f}")

    feasible = problem.satisfy(solution)
    st.sidebar.metric("Feasible", "✅ Yes" if feasible else "❌ No")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Problem Size:**")
    st.sidebar.write(f"- {len(problem.assembly_lines_list)} assembly lines")
    st.sidebar.write(f"- {len(problem.factories_list)} factories")
    st.sidebar.write(f"- {len(problem.engines_list)} engines")
    st.sidebar.write(f"- {len(problem.msns_list)} MSNs")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 JIT vs Optimized", "📊 Gantt Chart", "🎯 Analysis", "📉 KPI Evolution"])

    # ========================================================================
    # TAB 1: JIT vs Optimized Comparison
    # ========================================================================
    with tab1:
        st.header("Factory Load Comparison: JIT vs Optimized")

        jit_load = compute_jit_theoretical_load(problem)
        optimized_load = compute_optimized_load(problem, solution)
        switches, switch_details, transport_costs = count_factory_switches(problem, solution)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Factory Switches", f"{switches}")
            st.caption("Engines sent to non-preferred factories")

        with col2:
            st.metric("Total Travel Cost", f"€{kpis.get('travel_cost', 0):,.2f}")
            st.caption("Factory → FAL transport")

        # Comparison chart
        fig = create_comparison_chart(problem, jit_load, optimized_load)
        st.plotly_chart(fig, use_container_width=True)

        # Capacity utilization
        st.subheader("📊 Capacity Utilization Comparison")

        utilization_data = []
        for factory in problem.factories_list:
            jit_load_factory = jit_load.get(factory.id, {})
            opt_load_factory = optimized_load.get(factory.id, {})

            if factory.default_capacity is None:
                continue

            if jit_load_factory:
                jit_max = max(jit_load_factory.values())
                jit_peak_util = jit_max / factory.default_capacity * 100
            else:
                jit_max = 0
                jit_peak_util = 0

            if opt_load_factory:
                opt_max = max(opt_load_factory.values())
                opt_peak_util = opt_max / factory.default_capacity * 100
            else:
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

        # Switch details
        if switch_details:
            with st.expander(f"📋 Factory Switch Details ({len(switch_details)} switches)"):
                switch_df = pd.DataFrame(switch_details)
                st.dataframe(switch_df, use_container_width=True)

    # ========================================================================
    # TAB 2: Gantt Chart
    # ========================================================================
    with tab2:
        st.header("Engine Preparation Timeline")

        # Time slider
        current_time = st.slider(
            "⏱️ Time (days)",
            min_value=0,
            max_value=problem.time_horizon,
            value=0,
            step=1,
            help="Slide to see the system state at different times"
        )

        gantt_fig = create_gantt_chart(problem, solution, current_time)
        st.plotly_chart(gantt_fig, use_container_width=True)

        # Timeline details
        with st.expander("📋 Details at current time"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🏭 In Preparation")
                engines_in_prep = defaultdict(list)
                for engine in problem.engines_list:
                    if engine in solution.prep_start_times and engine in solution.engine_to_factory:
                        prep_start = solution.prep_start_times[engine]
                        prep_end = solution.prep_end_times[engine]
                        if prep_start <= current_time < prep_end:
                            factory = solution.engine_to_factory[engine]
                            engines_in_prep[factory.id].append(engine.id)

                if engines_in_prep:
                    for factory_id, engines in engines_in_prep.items():
                        st.write(f"**{factory_id}**: {', '.join(engines)}")
                else:
                    st.write("None")

            with col2:
                st.subheader("🚚 In Transit")
                engines_in_transit = []
                for engine in problem.engines_list:
                    if engine in solution.prep_end_times and engine in solution.engine_to_msn:
                        msn = solution.engine_to_msn[engine]
                        if msn is not None:  # Check if allocated
                            prep_end = solution.prep_end_times[engine]
                            arrival = solution.fal_arrival_times[engine]
                            if prep_end <= current_time < arrival:
                                factory = solution.engine_to_factory[engine]
                                engines_in_transit.append((engine.id, factory.id, msn.assembly_line_id))

                if engines_in_transit:
                    for eng_id, factory_id, fal_id in engines_in_transit:
                        st.write(f"{eng_id}: {factory_id} → {fal_id}")
                else:
                    st.write("None")

            with col3:
                st.subheader("✈️ Arrived")
                engines_arrived = defaultdict(list)
                for engine in problem.engines_list:
                    if engine in solution.fal_arrival_times and engine in solution.engine_to_msn:
                        msn = solution.engine_to_msn[engine]
                        if msn is not None:  # Check if allocated
                            arrival = solution.fal_arrival_times[engine]
                            if arrival <= current_time:
                                engines_arrived[msn.assembly_line_id].append(engine.id)

                if engines_arrived:
                    for fal_id, engines in engines_arrived.items():
                        st.write(f"**{fal_id}**: {len(engines)} engines")
                else:
                    st.write("None")

    # ========================================================================
    # TAB 3: Additional Analysis
    # ========================================================================
    with tab3:
        st.header("Additional Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💰 Transport Costs by Factory")
            if transport_costs:
                cost_df = pd.DataFrame([
                    {'Factory': factory_id, 'Transport Cost (€)': f"{cost:,.2f}"}
                    for factory_id, cost in sorted(transport_costs.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(cost_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("📦 Allocation Summary")
            allocated_engines = sum(1 for e in problem.engines_list if e in solution.engine_to_msn)
            unallocated_engines = len(problem.engines_list) - allocated_engines

            st.metric("Allocated Engines", f"{allocated_engines}/{len(problem.engines_list)}")
            st.metric("Unallocated Engines", f"{unallocated_engines}")

        # Engine type breakdown
        st.subheader("🔧 Engine Type Breakdown")
        engine_type_data = []
        for etype in set(e.engine_type for e in problem.engines_list):
            n_engines = len([e for e in problem.engines_list if e.engine_type == etype])
            n_msns = len([m for m in problem.msns_list if m.engine_type == etype])
            n_allocated = len([e for e in problem.engines_list
                             if e.engine_type == etype and e in solution.engine_to_msn])

            engine_type_data.append({
                'Engine Type': etype,
                'Total Engines': n_engines,
                'MSN Demand': n_msns,
                'Allocated': n_allocated,
                'Unallocated': n_engines - n_allocated,
                'Ratio': f"{n_engines/n_msns:.2f}" if n_msns > 0 else "N/A"
            })

        engine_df = pd.DataFrame(engine_type_data)
        st.dataframe(engine_df, use_container_width=True, hide_index=True)

        # Show unallocated engines if any
        unallocated = [e for e in problem.engines_list
                      if e not in solution.engine_to_msn or solution.engine_to_msn[e] is None]

        if unallocated:
            with st.expander(f"⚠️ Unallocated Engines ({len(unallocated)})"):
                unallocated_data = []
                for engine in unallocated:
                    unallocated_data.append({
                        'Engine ID': engine.id,
                        'Engine Type': engine.engine_type,
                        'Delivery Time': engine.delivery_time if engine.delivery_time is not None else 'Flexible',
                        'Baseline Factory': engine.baseline_factory_id,
                    })
                unallocated_df = pd.DataFrame(unallocated_data)
                st.dataframe(unallocated_df, use_container_width=True, hide_index=True)
                st.caption(f"These {len(unallocated)} engines could not be allocated to any MSN (likely due to shortage or timing constraints)")

    # ========================================================================
    # TAB 4: KPI Evolution
    # ========================================================================
    with tab4:
        st.header("KPI Evolution During Optimization")

        # Check if solution history is available
        if 'solution_history' not in st.session_state or not st.session_state.solution_history:
            st.warning("⚠️ No solution history available. Re-run the solver to collect intermediate solutions.")
            st.info("The solver stores all intermediate solutions found during optimization. This allows tracking how the objective values improve over time.")
            return

        solution_history = st.session_state.solution_history
        st.success(f"✅ Loaded {len(solution_history)} intermediate solutions")

        # Extract data for plotting
        iterations = [entry['index'] for entry in solution_history]
        fitness_values = [entry['fitness'] for entry in solution_history]

        # Extract KPI components
        kpi_names = list(solution_history[0]['kpis'].keys())
        kpi_data = {name: [entry['kpis'][name] for entry in solution_history] for name in kpi_names}

        # Plot 1: Overall Fitness Evolution
        st.subheader("📊 Overall Fitness Evolution")
        fig_fitness = go.Figure()
        fig_fitness.add_trace(go.Scatter(
            x=iterations,
            y=fitness_values,
            mode='lines+markers',
            name='Fitness',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=6)
        ))
        fig_fitness.update_layout(
            title="<b>Fitness Value Over Solutions</b>",
            xaxis_title="<b>Solution Index</b>",
            yaxis_title="<b>Fitness (lower is better)</b>",
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_fitness, use_container_width=True)

        # Plot 2: Individual KPI Components Evolution
        st.subheader("📈 Individual KPI Components")

        # Create subplot for all KPIs
        n_kpis = len(kpi_names)
        fig_kpis = make_subplots(
            rows=n_kpis,
            cols=1,
            subplot_titles=[f"<b>{name.replace('_', ' ').title()}</b>" for name in kpi_names],
            vertical_spacing=0.08,
            shared_xaxes=True
        )

        kpi_colors = {
            'baseline_changes': '#9b59b6',
            'arrival_delays': '#e74c3c',
            'travel_cost': '#3498db',
            'storage_cost': '#f39c12',
            'unallocated_msns': '#e67e22'
        }

        for idx, kpi_name in enumerate(kpi_names):
            row = idx + 1
            color = kpi_colors.get(kpi_name, '#95a5a6')

            fig_kpis.add_trace(
                go.Scatter(
                    x=iterations,
                    y=kpi_data[kpi_name],
                    mode='lines+markers',
                    name=kpi_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                    showlegend=False
                ),
                row=row, col=1
            )

            fig_kpis.update_yaxes(
                title_text=kpi_name.replace('_', ' ').title(),
                row=row, col=1,
                title_font=dict(size=11)
            )

        fig_kpis.update_xaxes(
            title_text="<b>Solution Index</b>",
            row=n_kpis, col=1,
            title_font=dict(size=13)
        )

        fig_kpis.update_layout(
            height=250 * n_kpis,
            title=dict(
                text="<b>KPI Components Evolution</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            hovermode='x unified',
            showlegend=False,
            template='plotly_white'
        )

        st.plotly_chart(fig_kpis, use_container_width=True)

        # Summary statistics
        st.subheader("📋 Improvement Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**First Solution vs Best Solution:**")
            improvement_data = []
            first_kpis = solution_history[0]['kpis']
            best_kpis = solution_history[-1]['kpis']

            for kpi_name in kpi_names:
                first_val = first_kpis[kpi_name]
                best_val = best_kpis[kpi_name]
                improvement = first_val - best_val
                improvement_pct = (improvement / first_val * 100) if first_val != 0 else 0

                improvement_data.append({
                    'KPI': kpi_name,
                    'First': f"{first_val:.2f}",
                    'Best': f"{best_val:.2f}",
                    'Improvement': f"{improvement:.2f}",
                    'Improvement (%)': f"{improvement_pct:.1f}%"
                })

            improvement_df = pd.DataFrame(improvement_data)
            st.dataframe(improvement_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**Overall Fitness:**")
            first_fitness = fitness_values[0]
            best_fitness = fitness_values[-1]
            fitness_improvement = first_fitness - best_fitness
            fitness_improvement_pct = (fitness_improvement / first_fitness * 100) if first_fitness != 0 else 0

            st.metric("First Fitness", f"{first_fitness:,.0f}")
            st.metric("Best Fitness", f"{best_fitness:,.0f}", delta=f"-{fitness_improvement:,.0f} ({fitness_improvement_pct:.1f}%)")
            st.metric("Total Solutions Found", len(solution_history))


if __name__ == "__main__":
    main()