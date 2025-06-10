import json
from typing import Tuple, Dict

import networkx as nx
from networkx import DiGraph
from datetime import datetime


def parse_DiGraph(json_path: str) -> Tuple[DiGraph, Dict[str, Tuple[float, int]], Dict[int, float]]:
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
        graph.add_node(node["id"],
                       acquisition_request_id=acquisition_requests_id,
                       chemin=[],
                       chemin_set=set(),
                       start=datetime.strptime(node["start_datetime"], '%Y-%m-%d %H:%M:%S.%f'),
                       end=datetime.strptime(node["end_datetime"], '%Y-%m-%d %H:%M:%S.%f'),
                       value_chemin=0)

    for node in nodes_obj:
        for succ in node['successor_node_ids']:
            graph.add_edge(node["id"], succ)

    solution = {}
    if "optimal_operational_solution" in obj:
        if obj["optimal_operational_solution"]["total_operational_score"]:
            solution["optimal"] = (obj["optimal_operational_solution"]["total_operational_score"],
                                   obj["optimal_number_of_acqrs_solution"]["number_of_acqrs"])

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
    return graph, solution, acquisition_dict


