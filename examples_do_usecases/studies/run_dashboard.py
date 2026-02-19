from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study import Hdf5Database

# study_name = "rcpsp-study-cpsat-vs-hexaly"
# study_name = "fjsp-study-cpsat-vs-hexaly"
# study_name = "rcpsp-study-cpsat-vs-hexaly-vs-greedy"
# study_name = "jsp-study-cpsat-vs-hexaly-vs-optalcp-vs-tempo-vs-cpo-"
# study_name = "fjsp-study-cpsat-vs-hexaly-vs-optal"
# study_name = "jsp-study-cpsat-vs-hexaly-vs-optalcp-vs-tempo-vs-cpo-2-"
# study_name = "rcpsp-study-cpsat-vs-hexaly-vs-optal-greedy-sm"
# study_name = "tsp-tw-study-solomon-pesant"
# study_name = "tsp-tw-study-dumas"
# study_name = "tsp-tw-study-ohlmann"
# study_name = "spp-study"
# study_name = "msrcpsp"
study_name = "sm-study"
# study_name = "workforce-scheduling-study-0"
study_name = "rcpsp-study-cpsat-vs-hexaly-vs-optal-greedy-sm"
study_name = "jsp-study-cpsat-vs-hexaly-vs-optalcp-vs-tempo-vs-cpo-"
study_name = "workforce-scheduling-study-0"
# retrieve data
with Hdf5Database(f"{study_name}.h5") as database:
    results = database.load_results()
    if True:
        l = []
        for r in results:
            if "fit" in r.columns:
                if all(x is None for x in r.fit):
                    # print('continue')
                    continue
                if min(r.fit) < 0:
                    r.fit = -r.fit
                l.append(r)
        results = l



# launch dashboard with this data
app = Dashboard(results=results)
app.run()