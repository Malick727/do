import datetime
import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
import pandas as pd
import time
import json_transformations
from streamlit_local_storage import LocalStorage


ls = LocalStorage()

from PIL import Image
# Disable "Decompression Bomb" protection for large maps
Image.MAX_IMAGE_PIXELS = None


# Path Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)

# Imports
try:
    import runners
    from discrete_optimization.generic_tools.cp_tools import ParametersCp
    from multibatching.utils import plot_solution_graph, generate_multibatching_problem, plot_solution_on_map_by_transport
    from multibatching.parse_config_data import parse_json_to_problem
    from examples_do_usecases.multibatching.instance import init_instance
    from logger import StreamlitLogger    

except ImportError as e:
    st.error("Critical Import Error.")
    st.error(f"Details: {e}")
    st.stop()

data = None

def parse_problem(fix_problem: bool = False):
    config_file = os.path.join(root_dir, "multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d)
    if fix_problem:
        problem.products[3].valid_transports = set(problem.transport_types)
    return problem

# Monkey Patches
original_show = plt.show
def dummy_show(*args, **kwargs): pass
plt.show = dummy_show

def streamlit_plotly_show(self, *args, **kwargs):
    st.plotly_chart(self, use_container_width=True)
go.Figure.show = streamlit_plotly_show

# Sanitizer
def sanitize_solution_data(solution):
    """
    Fixes NoneType names and empty locations to prevent plotting crashes.
    """
    coords_map = {}
    path_folder = "multibatching/datalocal"
    path_file = os.path.join(path_folder, "coordinates.json")

    if os.path.exists(path_file):
        try:
            with open(path_file, "r") as f:
                coords_map = json.load(f)
        except:
            coords_map = {}

    if solution.list_flows:
        for flow in solution.list_flows:
            link = flow.transport_link
            if not link: continue
            
            tt = link.transport_type
            if tt.name is None:
                tt.name = "Standard_Transport" 

            for loc in [link.location_l1, link.location_l2]:
                if not loc.name:
                    loc.name = f"Location_{loc.id}"
                
                if loc.name not in coords_map:
                    try:
                        numeric_id = float(loc.id)
                    except (ValueError, TypeError):
                        numeric_id = float(hash(loc.id) % 100)
                    
                    offset = numeric_id * 0.1
                    coords_map[loc.name] = [48.0 + offset, 10.0 + offset]

    os.makedirs(path_folder, exist_ok=True)
    with open(path_file, "w") as f:
        json.dump(coords_map, f)

    return solution


# --- SOLVER STRATEGIES ---
strategies = {
    "Networkx": runners.run_networkx,
    "OpenEvolve": runners.run_openevolve,
    "ColumnGeneration": runners.run_column_generation,
    "ASP": runners.run_asp,
    "Flow + PostPro ": runners.run_flow_postpro,
    "Flow + PostPro + Complete": runners.run_flow_postpro_complete,
    "Flow (Single Batch) + Greedy": runners.run_flow_greedy,
    "Benders (Single batch) + Greedy": runners.run_benders_greedy,
    "GreedyPacking": runners.run_greedy_packing,
    "Milp single": runners.run_milp_single,
    "Flow Model (Fast)": runners.run_flow_model,
    "Unit Model (Detailed)": runners.run_unit_model,
    "Benders Decomposition": runners.run_benders,
    "CPMPY": runners.run_cpmpy
}

# --- DISPATCHER FUNCTION ---
def run_solver_strategy(strategy, problem, p, time_limit, **kwargs):
    """
    Dispatches the execution to the correct runner function based on strategy name.
    """
    
    runner = strategies.get(strategy)
    
    if runner:
        # Pass args dynamically
        return runner(problem=problem, p=p, time_limit=time_limit, **kwargs)
    else:
        print(f"Error: Strategy '{strategy}' not implemented.")
        return None


# --- UI CONFIGURATION ---
st.set_page_config(page_title="Multibatching Solver", layout="wide")

# --- INITIALIZE HISTORY STATE ---
if "history" not in st.session_state:
    st.session_state["history"] = []

try:
    stored_history_json = ls.getItem("solver_history")
    if stored_history_json:
        if not st.session_state["history"]:
            st.session_state["history"] = json.loads(stored_history_json)
except Exception as e:
    print(f"Error loading localStorage: {e}")

st.title("🚚 Multibatching Solver Dashboard")

# --- SIDEBAR ---
st.sidebar.header("Configuration")


strategy = st.sidebar.selectbox(
    "Optimization Strategy", 
    strategies.keys()
)

st.sidebar.divider()
st.sidebar.subheader("Problem settings")
use_generated_problem = st.sidebar.toggle("Generate Problem", value=False)
gen_params = {}

if use_generated_problem:
    use_demo = st.sidebar.toggle("Use Demo Instance", value=True)
    if not use_demo:
        st.sidebar.subheader("Random Generator Parameters")
        with st.sidebar.expander("1. Dimensions & General", expanded=True):
            gen_params['num_locations'] = st.number_input("num_locations", value=6, min_value=2)
            gen_params['num_transport_types'] = st.number_input("num_transport_types", value=2, min_value=1)
            gen_params['num_products'] = st.number_input("num_products", value=2, min_value=1)
            
            use_fixed_seed = st.checkbox("Use Fixed Seed", value=True)
            gen_params['seed'] = st.number_input("seed", value=42) if use_fixed_seed else None

        with st.sidebar.expander("2. Distances & Capacity", expanded=False):
            c1, c2 = st.columns(2)
            gen_params['min_dist'] = c1.number_input("min_dist", value=1)
            gen_params['max_dist'] = c2.number_input("max_dist", value=10)
            c3, c4 = st.columns(2)
            gen_params['min_capacity'] = c3.number_input("min_capacity", value=10)
            gen_params['max_capacity'] = c4.number_input("max_capacity", value=20)
            gen_params['max_links_per_pair'] = st.number_input("max_links_per_pair", value=2, min_value=1)

        with st.sidebar.expander("3. Products & Demand", expanded=False):
            c1, c2 = st.columns(2)
            gen_params['min_product_size'] = c1.number_input("min_product_size", value=1)
            gen_params['max_product_size'] = c2.number_input("max_product_size", value=5)
            c3, c4 = st.columns(2)
            gen_params['min_product_value'] = c3.number_input("min_product_value", value=100)
            gen_params['max_product_value'] = c4.number_input("max_product_value", value=1000)
            gen_params['max_demand_abs'] = st.number_input("max_demand_abs", value=50)
            gen_params['demand_sparsity'] = st.slider("demand_sparsity", 0.0, 1.0, 0.5)

        with st.sidebar.expander("4. Transport Costs & Emission", expanded=False):
            c1, c2 = st.columns(2)
            gen_params['min_tt_cost'] = c1.number_input("min_tt_cost", value=10)
            gen_params['max_tt_cost'] = c2.number_input("max_tt_cost", value=100)
            c3, c4 = st.columns(2)
            gen_params['min_tt_speed'] = c3.number_input("min_tt_speed", value=0.1, step=0.1)
            gen_params['max_tt_speed'] = c4.number_input("max_tt_speed", value=1.0, step=0.1)
            c5, c6 = st.columns(2)
            gen_params['min_tt_emissions'] = c5.number_input("min_tt_emissions", value=10)
            gen_params['max_tt_emissions'] = c6.number_input("max_tt_emissions", value=100)
    else:
        st.sidebar.info("Using demo instance")
else:
    upload_json_file = st.sidebar.toggle("Upload problem as JSON file", value=False)
    if upload_json_file:
        uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
        if uploaded_file is not None:
            data = json.load(uploaded_file)
    else:
        st.sidebar.info("Using local config-local.json")

st.sidebar.divider()
time_limit = st.sidebar.number_input("Time Limit (seconds)", value=30, min_value=5)

if st.sidebar.button("🗑️ Clear Comparison History"):
    st.session_state["history"] = []
    ls.deleteItem("solver_history")
    st.sidebar.success("History cleared!")

# Json generation handler
def get_generated_json():
    if use_generated_problem:
        if use_demo:
            print("Loading Demo Instance...")
            problem = init_instance()
        else:
            print(f"Generating Random Instance (Seed: {gen_params.get('seed')})...")
            problem = generate_multibatching_problem(**gen_params)
    else:
        print("Parsing problem from local JSON...")
        problem = parse_problem()
    
    return json_transformations.export_multibatching_problem_to_json(problem, "problem_from_streamlit.json")


# --- MAIN EXECUTION ---

tab1, tab2 = st.tabs(["🗺️ Solver", "📈 Comparison"])

with tab1:
    if st.button("🚀 Run Solver", type="primary"):
        
        st.subheader("⚙️ Execution in Progress")
        with st.container(height=300, border=True):
            log_container = st.empty()
        
        logger = StreamlitLogger(log_container)
        old_stdout = sys.stdout 
        sys.stdout = logger

        start_time = time.time() 

        try:
            print(f"--- Starting process with strategy: {strategy} ---")
            
            if use_generated_problem:
                if use_demo:
                    print("Loading Demo Instance...")
                    problem = init_instance()
                else:
                    print(f"Generating Random Instance (Seed: {gen_params.get('seed')})...")
                    problem = generate_multibatching_problem(**gen_params)
            else:
                if upload_json_file:
                    problem = json_transformations.load_multibatching_problem_from_json(data)
                else:
                    print("Parsing problem from local JSON...")
                    problem = parse_problem()

            print(f"Problem Loaded: {len(problem.locations)} locations, {len(problem.transport_types)} transport types.")

            # Shared Parameters
            p = ParametersCp.default_cpsat()
            p.nb_process = 8
            kwargs = {"log_search_progress": True} 

            # --- CALL DISPATCHER ---
            res = run_solver_strategy(
                strategy=strategy,
                problem=problem,
                p=p,
                time_limit=time_limit,
                **kwargs
            )

            # Finalize Logs
            log_content = logger.get_content()
            log_container.empty()
            
            elapsed_time = time.time() - start_time

            if res:
                sol = res[-1][0]
                kpis = problem.evaluate(sol)
                is_valid = problem.satisfy(sol)

                # History Record
                run_record = {
                    "Strategy": strategy,
                    "Time (s)": round(elapsed_time, 2),
                    "Transport Cost": round(kpis.get('transport', 0), 2),
                    "Emissions": round(kpis.get('emission', 0), 2),
                    "Capital": round(kpis.get('capital', 0), 2),
                    "Valid": is_valid
                }
                st.session_state["history"].append(run_record)

                try:
                    history_json = json.dumps(st.session_state["history"])
                    ls.setItem("solver_history", history_json)
                    print(">> History saved to LocalStorage.")
                except Exception as e:
                    print(f"Error saving to LocalStorage: {e}")

                st.success(f"Solution Found in {elapsed_time:.2f}s!")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Transport Cost", f"{kpis.get('transport',0):.1f}")
                c2.metric("Emissions", f"{kpis.get('emission',0):.1f}")
                c3.metric("Capital Cost", f"{kpis.get('capital',0):.1f}")
                c4.metric("Valid?", "✅ Yes" if is_valid else "❌ No")

                st.divider()

                st.caption("Visualizing route and batching...")
                plt.clf() 
                try:
                    if use_generated_problem or upload_json_file:
                        plot_solution_graph(problem=problem, solution=sol)
                        st.pyplot(plt)
                    
                    sol = sanitize_solution_data(sol)
                    plot_solution_on_map_by_transport(solution=sol)
                except Exception as e:
                    st.warning(f"Could not render the graph: {e}")

            else:
                st.error("No feasible solution found.")
                with st.expander("View Error Logs", expanded=True):
                    st.text_area("Error Logs", value=log_content, height=300)

        except Exception as e:
            st.error("An error occurred during execution:")
            st.exception(e)
            st.subheader("Logs before crash:")
            st.text_area("Crash Logs", value=logger.get_content(), height=300)

        finally:
            sys.stdout = old_stdout

    # Download current problem instance
    st.download_button("Download problem instance as a JSON file", 
                       type="secondary", 
                       data=get_generated_json(), 
                       file_name="instance" + datetime.datetime.now().strftime("%d%m%Y%H%M%S") + ".json", 
                       mime="application/json")

with tab2:
    # Comparison Tab
    print("History in Session State: ", st.session_state["history"])
    if st.session_state["history"]:
        st.header("📊 Multi-Run Comparison")
        
        df_history = pd.DataFrame(st.session_state["history"])
        st.dataframe(df_history, use_container_width=True)
        
        if not df_history.empty:
            st.subheader("Cost Comparison")
            st.bar_chart(df_history, x="Strategy", y="Transport Cost", color="Strategy")

