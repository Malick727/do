import json
from typing import Tuple, Dict

import networkx as nx
from networkx import DiGraph
from datetime import datetime

from satellite_path_planning.problem import SppProblem


def parse_spp_problem(json_path: str) -> SppProblem:
    """
    @param json_path: the file to parse
    @return: the graph who modelize the problem, solutions already compute and the number of acquisition
    """
    obj = json.load(open(json_path, "r"))
    nodes_obj = obj["nodes"]
    acquisition_requests = obj["acquisition_requests"]
    acquisition_dict = {x["id"]: x["operational_score"] for x in acquisition_requests}

    graph = nx.DiGraph()

    for node in nodes_obj:
        acquisition_requests_id = node['acquisition_request_id']
        if acquisition_requests_id is None:
            if len(node['successor_node_ids']) == 0:
                acquisition_requests_id = "final"
            else:
                acquisition_requests_id = "first"
        if node["start_datetime"] is None:
            if acquisition_requests_id == "first":
                start = datetime(day=1, month=1, year=1970)
            if acquisition_requests_id == "final":
                start = datetime(day=1, month=1, year=2070)
        else:
            start = datetime.strptime(node["start_datetime"], '%Y-%m-%d %H:%M:%S.%f')
        if node["end_datetime"] is None:
            if acquisition_requests_id == "first":
                end = datetime(day=1, month=1, year=1970)
            if acquisition_requests_id == "final":
                end = datetime(day=1, month=1, year=2070)
        else:
            end = datetime.strptime(node["end_datetime"], '%Y-%m-%d %H:%M:%S.%f')
        graph.add_node(node["id"],
                       acquisition_request_id=acquisition_requests_id,
                       chemin=[],
                       value=acquisition_dict.get(acquisition_requests_id, 0),
                       chemin_set=set(),
                       start=start,
                       end=end,
                       value_chemin=0)

    for node in nodes_obj:
        for succ in node['successor_node_ids']:
            graph.add_edge(node["id"], succ)

    solution = {}
    if "optimal_operational_solution" in obj:
        if obj["optimal_operational_solution"]["total_operational_score"]:
            solution["optimal"] = (obj["optimal_operational_solution"]["total_operational_score"],
                                   obj["optimal_number_of_acqrs_solution"]["number_of_acqrs"])
    if "ramp_operational_solution" in obj:
        if obj["ramp_operational_solution"]["total_operational_score"]:
            try:
                solution["ramp"] = (obj["ramp_operational_solution"]["total_operational_score"],
                                    obj["ramp_number_of_acqrs_solution"]["number_of_acqrs"])
            except:
                solution["ramp"] = (obj["ramp_operational_solution"]["total_operational_score"],
                                    len(obj["ramp_operational_solution"]["acqr_id_chain"]))

    if obj["greedy_operational_solution"]["total_operational_score"]:
        try:
            solution["greedy"] = (obj["greedy_operational_solution"]["total_operational_score"],
                                  obj["greedy_number_of_acqrs_solution"]["number_of_acqrs"])
        except:
            solution["greedy"] = (obj["greedy_operational_solution"]["total_operational_score"],
                                  len(obj["greedy_operational_solution"]["acqr_id_chain"]))
    problem = SppProblem(graph=graph,
                         acquisition_dict=acquisition_dict)
    problem.solution = solution
    return problem


