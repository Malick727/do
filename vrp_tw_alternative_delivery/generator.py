import math
import random
from typing import List, Dict, Tuple
import numpy as np
from discrete_optimization.vrptw.problem import VRPTWProblem
from vrp_tw_alternative_delivery.problem import ProductInstance, Vehicle, VrpTwAlternativeDeliveryProblem, ProductType, \
    Node, RequirementRelation


def generate_from_vrptw(
        vrptw_problem: VRPTWProblem,
        nb_nodes_target: int,
        nb_vehicles_target: int
) -> VrpTwAlternativeDeliveryProblem:
    """
    Generates a VrpTwAlternativeDeliveryProblem from a VRPTWProblem instance.
    """

    # --- 1. Define Product Types and Requirements ---

    # Renewable/immaterial product
    pt_renewable = ProductType(id="pt_renew_1", name="Special Service", is_renewable=True)

    # Non-renewable/actual products
    pt_non_renew_1 = ProductType(id="pt_nonrenew_1", name="Standard Goods", is_renewable=False)
    pt_non_renew_2 = ProductType(id="pt_nonrenew_2", name="Special Goods", is_renewable=False)

    # Requirements: "Special Goods" require "Special Service"
    relation = RequirementRelation(product_type_1=pt_non_renew_2, product_type_2=pt_renewable)

    # --- 2. Select Subset of Nodes ---
    if nb_nodes_target > vrptw_problem.nb_nodes:
        nb_nodes_target = vrptw_problem.nb_nodes

    # Select depot + (nb_nodes_target - 1) random customers
    selected_customers = random.sample(vrptw_problem.customers, nb_nodes_target - 1)
    selected_nodes_old_indices = [vrptw_problem.depot_node] + selected_customers

    # Create mapping from old -> new indices
    old_to_new_index_map = {old_id: new_id for new_id, old_id in enumerate(selected_nodes_old_indices)}
    new_to_old_index_map = {v: k for k, v in old_to_new_index_map.items()}

    # --- 3. Create New Node List ---
    new_nodes: List[Node] = []
    for new_id in range(nb_nodes_target):
        old_id = new_to_old_index_map[new_id]

        old_tw = vrptw_problem.time_windows[old_id]
        new_tw_list = [(int(old_tw[0]), int(old_tw[1]))]  # List of tuples + cast to int

        new_service_time = int(vrptw_problem.service_times[old_id])
        is_depot = (old_id == vrptw_problem.depot_node)

        new_node = Node(
            id=new_id,
            name=f"Node {new_id}" + (" (Depot)" if is_depot else ""),
            is_client=not is_depot,
            is_depot=is_depot,
            time_window=new_tw_list[0],
            service_time=new_service_time
        )
        new_nodes.append(new_node)

    depot_node_new = new_nodes[0]

    # --- 4. Create New Matrices (int) ---
    new_dist_matrix = np.zeros((nb_nodes_target, nb_nodes_target), dtype=int)
    for i in range(nb_nodes_target):
        for j in range(nb_nodes_target):
            old_i = new_to_old_index_map[i]
            old_j = new_to_old_index_map[j]
            new_dist_matrix[i, j] = int(vrptw_problem.distance_matrix[old_i, old_j])

    # For this problem, assume travel time and cost = distance
    new_travel_times = new_dist_matrix.copy()
    new_travel_costs = new_dist_matrix.copy()

    # --- 5. Define Client Needs based on VRPTW demands ---
    client_needs_spec: Dict[int, List[Tuple[ProductType, int]]] = {}
    total_needs: Dict[ProductType, int] = {}

    for new_id in range(1, nb_nodes_target):  # Iterate over clients (skip depot)
        old_id = new_to_old_index_map[new_id]
        demand_qty = 1 # int(vrptw_problem.demands[old_id])
        if demand_qty <= 0:
            demand_qty = 1  # Ensure everyone needs at least 1 unit

        # Randomly assign if they need "Standard" or "Special" goods
        if random.random() < 0.3:  # 30% chance of needing special goods
            pt_need = pt_non_renew_2
        else:
            pt_need = pt_non_renew_1

        client_needs_spec[new_id] = [(pt_need, demand_qty)]
        total_needs[pt_need] = total_needs.get(pt_need, 0) + demand_qty

    # --- 6. Create Vehicles and Load them ---
    new_vehicles: List[Vehicle] = []

    # Master list to ensure unique product instance IDs
    product_instances_db: List[ProductInstance] = []

    # Create the single renewable instance
    renew_instance = ProductInstance(id="P_RENEW_0", name="Special Service Unit", product_type=pt_renewable)
    product_instances_db.append(renew_instance)

    total_need_std = total_needs.get(pt_non_renew_1, 0)
    total_need_spe = total_needs.get(pt_non_renew_2, 0)

    # Create a heterogeneous fleet
    nb_special_vehicles = max(1, nb_vehicles_target // 2)
    nb_standard_vehicles = nb_vehicles_target - nb_special_vehicles

    # Calculate load per vehicle (total need / nb_vehicles) + buffer
    load_per_std_vehicle = math.ceil(total_need_std / max(1, nb_standard_vehicles)) + 2
    load_per_spe_vehicle = math.ceil(total_need_spe / max(1, nb_special_vehicles)) + 2

    v_id = 0
    # Create Standard Vehicles
    for _ in range(nb_standard_vehicles):
        vehicle_load: List[ProductInstance] = []
        for _ in range(load_per_std_vehicle):
            inst_id = f"P_NR1_{v_id}_{_}"
            inst = ProductInstance(id=inst_id, name=f"Std Good {inst_id}", product_type=pt_non_renew_1)
            vehicle_load.append(inst)
            product_instances_db.append(inst)

        new_v = Vehicle(
            id=v_id, name=f"Standard Vehicle {v_id}",
            load=vehicle_load,
            min_speed=1.0, max_speed=1.0,
            start_node=depot_node_new, end_node=depot_node_new
        )
        new_vehicles.append(new_v)
        v_id += 1

    # Create Special Vehicles
    for _ in range(nb_special_vehicles):
        vehicle_load: List[ProductInstance] = []
        # Add the renewable product!
        inst_id = f"P_RENEW_{v_id}"
        vehicle_load.append(ProductInstance(id=inst_id,
                                            name=f"Special Service Unit {v_id}",
                                            product_type=pt_renewable))
        # Add the non-renewable special goods
        for _ in range(load_per_spe_vehicle):
            inst_id = f"P_NR2_{v_id}_{_}"
            inst = ProductInstance(id=inst_id, name=f"Spe Good {inst_id}", product_type=pt_non_renew_2)
            vehicle_load.append(inst)
            product_instances_db.append(inst)
        new_v = Vehicle(
            id=v_id, name=f"Special Vehicle {v_id}",
            load=vehicle_load,
            min_speed=1.0,
            max_speed=1.0,
            start_node=depot_node_new, end_node=depot_node_new
        )
        new_vehicles.append(new_v)
        v_id += 1

    # --- 7. Instantiate the new problem ---

    problem = VrpTwAlternativeDeliveryProblem(
        nodes=new_nodes,
        vehicles=new_vehicles,
        distance_matrix=new_dist_matrix,
        travel_times=new_travel_times,
        travel_costs=new_travel_costs,
        requirement_relations=[relation],
        # client_needs=client_needs_spec
    )
    return problem
