from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study import Hdf5Database
from discrete_optimization.generic_tools.study.experiment import CONFIG, INSTANCE, CONFIG_PREFIX
from collections import defaultdict

study_name = "multibatching-airbus-final_2 copy"

# 1. Retrieve data
with Hdf5Database(f"{study_name}.h5") as database:
    results = database.load_results()

# 2. Modify config names to include a counter per instance
# This creates "unique" solvers for each attempt (e.g., ConfigA_#1, ConfigA_#2)
# while keeping the Instance name identical.
config_counts = defaultdict(int)
# We track counts per (Config, Instance) to identify which attempt number it is
run_tracker = defaultdict(int)

for df in results:

    conf_name = df.attrs["config"]["name"]
    inst_name = df.attrs.get(INSTANCE, "unknown")

    # Track which run number this is for this specific Config + Instance combo
    run_tracker[(conf_name, inst_name)] += 1
    run_nb = run_tracker[(conf_name, inst_name)]

    # Update the CONFIG name to include the run number
    # This treats each of your 3 runs as a separate "competitor" in the dashboard
    df.attrs[INSTANCE] = f"instance_run{run_nb}"
print(set([df.attrs["config"]["name"] for df in results]))
# 3. Launch dashboard
# The dashboard will now show 120 configs (40 configs * 3 runs)
# and your original number of instances.
app = Dashboard(results=results)
app.run()