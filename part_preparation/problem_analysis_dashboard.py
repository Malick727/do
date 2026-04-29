"""
Interactive Problem Analysis Dashboard

Analyzes capacity constraints and bottlenecks in the engine preparation problem.
Shows theoretical JIT production load vs factory capacity.

Run with: uv run streamlit run part_preparation/problem_analysis_dashboard.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import defaultdict
import numpy as np


def compute_jit_production_load(problem):
    """
    Compute theoretical JIT production load.

    For each MSN, computes when prep must happen at the preferred factory
    to have the engine arrive EXACTLY at the installation date (pure JIT).

    Returns:
        Dict[factory_id -> Dict[time -> n_engines_in_prep]]
    """
    factory_load = defaultdict(lambda: defaultdict(int))

    for msn in problem.msns_list:
        # Find preferred factory (matches MSN's engine type and FAL)
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

        # Get actual preparation duration from factory data
        prep_duration = factory.get_preparation_duration(msn.engine_type)
        travel_time = problem.get_travel_time(factory, problem.assembly_lines[msn.assembly_line_id])

        # Work backwards from installation date (pure JIT)
        arrival_time = msn.engine_installation_date  # Arrive exactly when needed
        prep_end = arrival_time - travel_time        # Finish prep before travel
        prep_start = prep_end - prep_duration         # Start prep
        print("Prep start", prep_start, prep_end)
        # Record load
        for t in range(max(0, prep_start), min(problem.time_horizon, prep_end)):
            factory_load[factory.id][t] += 1

    return factory_load


def analyze_bottlenecks(problem, jit_load):
    """Identify bottleneck periods where JIT load exceeds capacity"""
    bottlenecks = []

    for factory in problem.factories_list:
        if factory.default_capacity is None:
            continue

        load = jit_load.get(factory.id, {})

        for t, n_engines in load.items():
            if n_engines > factory.default_capacity:
                bottlenecks.append({
                    'factory': factory.id,
                    'time': t,
                    'load': n_engines,
                    'capacity': factory.default_capacity,
                    'overflow': n_engines - factory.default_capacity,
                    'engine_type': list(factory.compatible_engine_types)[0],
                    'fal': factory.default_assembly_line.id
                })

    return pd.DataFrame(bottlenecks)


def create_capacity_analysis_chart(problem, jit_load, selected_factories=None, show_cumulative=False):
    """Create interactive capacity analysis chart"""

    if selected_factories is None:
        selected_factories = [f.id for f in problem.factories_list]

    # Filter factories
    factories_to_plot = [f for f in problem.factories_list if f.id in selected_factories]

    # Create subplots
    n_factories = len(factories_to_plot)
    fig = make_subplots(
        rows=n_factories,
        cols=1,
        subplot_titles=[f"<b>{f.id}</b> (Capacity={f.default_capacity}, FAL={f.default_assembly_line.id})"
                       for f in factories_to_plot],
        vertical_spacing=0.05,
        shared_xaxes=True,
    )

    for idx, factory in enumerate(factories_to_plot):
        row = idx + 1

        # Get load timeline
        load = jit_load.get(factory.id, {})
        times = sorted(load.keys()) if load else []

        if show_cumulative:
            # Cumulative production
            loads = [sum(load[t] for t in times if t <= time) for time in times]
            y_label = "Cumulative Engines"
        else:
            # Instantaneous load
            loads = [load[t] for t in times]
            y_label = "Engines in Prep"

        # JIT load line
        fig.add_trace(
            go.Scatter(
                x=times,
                y=loads,
                mode='lines',
                name='JIT Load',
                line=dict(color='#3498db', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.3)',
                showlegend=(idx == 0),
            ),
            row=row, col=1
        )

        if not show_cumulative and factory.default_capacity is not None:
            # Capacity line (only for instantaneous view)
            fig.add_trace(
                go.Scatter(
                    x=[0, problem.time_horizon],
                    y=[factory.default_capacity, factory.default_capacity],
                    mode='lines',
                    name='Capacity',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=(idx == 0),
                ),
                row=row, col=1
            )

            # Highlight bottleneck periods
            bottleneck_periods = [(t, load[t]) for t in times if load[t] > factory.default_capacity]
            if bottleneck_periods:
                bt, bl = zip(*bottleneck_periods)
                fig.add_trace(
                    go.Scatter(
                        x=bt,
                        y=bl,
                        mode='markers',
                        name='Bottleneck',
                        marker=dict(color='red', size=8, symbol='x'),
                        showlegend=(idx == 0),
                    ),
                    row=row, col=1
                )

        # Update y-axis
        fig.update_yaxes(
            title_text=y_label,
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
            text="<b>Factory Capacity Analysis - JIT Production Load</b>",
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


def main():
    st.set_page_config(
        page_title="Problem Analysis Dashboard",
        page_icon="🏭",
        layout="wide"
    )

    st.title("🏭 Engine Preparation Problem - Capacity Analysis")

    # Check if problem is loaded
    if 'problem' not in st.session_state:
        st.warning("⚠️ No problem instance loaded.")
        st.info("""
        To use this dashboard:
        1. Generate a problem instance
        2. Save it to session state: `st.session_state.problem = problem`
        3. Or run via the instance generator with dashboard launch

        Example:
        ```python
        from part_preparation.instance_generator import generate_instance, LARGE_INSTANCE
        problem = generate_instance(LARGE_INSTANCE)

        import streamlit as st
        st.session_state.problem = problem
        ```
        """)
        return

    problem = st.session_state.problem

    # Sidebar controls
    st.sidebar.header("⚙️ Analysis Parameters")

    st.sidebar.info("""
    **JIT Production Load**:
    Shows theoretical load if engines are prepared Just-In-Time
    to arrive EXACTLY at installation date.

    Uses actual factory preparation durations and travel times.
    """)

    # View mode
    show_cumulative = st.sidebar.checkbox(
        "Show Cumulative Production",
        value=False,
        help="Show cumulative engines produced over time instead of instantaneous load"
    )

    # Factory filter by engine type
    engine_types_flat = list(set(
        etype
        for f in problem.factories_list
        for etype in f.compatible_engine_types
    ))

    selected_engine_types = st.sidebar.multiselect(
        "Filter by Engine Type",
        options=engine_types_flat,
        default=engine_types_flat,
        help="Show only factories producing these engine types"
    )

    # Factory filter by FAL
    fal_ids = [fal.id for fal in problem.assembly_lines_list]
    selected_fals = st.sidebar.multiselect(
        "Filter by FAL",
        options=fal_ids,
        default=fal_ids,
        help="Show only factories assigned to these FALs"
    )

    # Compute JIT load (pure JIT - no configurable lead time)
    jit_load = compute_jit_production_load(problem)

    # Filter factories
    selected_factories = [
        f.id for f in problem.factories_list
        if any(etype in selected_engine_types for etype in f.compatible_engine_types)
        and f.default_assembly_line.id in selected_fals
    ]

    # Summary metrics
    st.header("📊 Problem Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("FALs", len(problem.assembly_lines_list))
        for fal in problem.assembly_lines_list:
            msns_count = len([m for m in problem.msns_list if m.assembly_line_id == fal.id])
            st.caption(f"{fal.id}: takt={fal.takt_pace}d, {msns_count} MSNs")

    with col2:
        st.metric("MSNs", len(problem.msns_list))
        for etype in engine_types_flat:
            msns_count = len([m for m in problem.msns_list if m.engine_type == etype])
            st.caption(f"{etype}: {msns_count}")

    with col3:
        st.metric("Engines", len(problem.engines_list))
        for etype in engine_types_flat:
            engines_count = len([e for e in problem.engines_list if e.engine_type == etype])
            st.caption(f"{etype}: {engines_count}")

    with col4:
        st.metric("Factories", len(problem.factories_list))
        for etype in engine_types_flat:
            factories_count = len([f for f in problem.factories_list if etype in f.compatible_engine_types])
            st.caption(f"{etype}: {factories_count}")

    # Bottleneck analysis
    st.header("🚨 Bottleneck Analysis")

    bottlenecks_df = analyze_bottlenecks(problem, jit_load)

    if len(bottlenecks_df) > 0:
        # Summary
        n_bottleneck_factories = bottlenecks_df['factory'].nunique()
        total_overflow = bottlenecks_df['overflow'].sum()
        peak_overflow = bottlenecks_df['overflow'].max()

        col1, col2, col3 = st.columns(3)
        col1.metric("Factories with Bottlenecks", f"{n_bottleneck_factories}/{len(problem.factories_list)}")
        col2.metric("Total Engine-Days Overflow", f"{total_overflow:.0f}")
        col3.metric("Peak Overflow", f"{peak_overflow:.0f} engines")

        # Bottleneck details
        with st.expander("📋 Bottleneck Details", expanded=False):
            st.dataframe(
                bottlenecks_df.sort_values('overflow', ascending=False).head(20),
                use_container_width=True
            )
    else:
        st.success("✅ No bottlenecks detected with current lead time!")

    # Capacity analysis chart
    st.header("📈 Factory Capacity Timeline")

    if selected_factories:
        fig = create_capacity_analysis_chart(
            problem,
            jit_load,
            selected_factories=selected_factories,
            show_cumulative=show_cumulative
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No factories match the selected filters.")

    # Capacity utilization summary
    st.header("📊 Capacity Utilization Summary")

    utilization_data = []
    for factory in problem.factories_list:
        load = jit_load.get(factory.id, {})
        if not load or factory.default_capacity is None:
            continue

        avg_load = np.mean(list(load.values()))
        max_load = max(load.values())
        avg_utilization = avg_load / factory.default_capacity * 100
        peak_utilization = max_load / factory.default_capacity * 100

        utilization_data.append({
            'Factory': factory.id,
            'Engine Type': list(factory.compatible_engine_types)[0],
            'FAL': factory.default_assembly_line.id,
            'Capacity': factory.default_capacity,
            'Avg Load': f"{avg_load:.1f}",
            'Max Load': max_load,
            'Avg Utilization (%)': f"{avg_utilization:.1f}",
            'Peak Utilization (%)': f"{peak_utilization:.1f}",
            'Status': '🚨 Bottleneck' if peak_utilization > 100 else '✅ OK'
        })

    if utilization_data:
        util_df = pd.DataFrame(utilization_data)
        st.dataframe(
            util_df.sort_values('Peak Utilization (%)', ascending=False),
            use_container_width=True,
            hide_index=True
        )


if __name__ == "__main__":
    main()