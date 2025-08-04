from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study import Hdf5Database

study_name = "rcpsp-study-cpsat-vs-hexaly"
study_name = "fjsp-study-cpsat-vs-hexaly"
study_name = "spp-study_"


# retrieve data
with Hdf5Database(f"{study_name}.h5") as database:
    results = database.load_results()
    results = results


# launch dashboard with this data
app = Dashboard(results=results)
app.run()