import json
from dataclasses import asdict
from typing import Union
import math

import sys
import os

# Path Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)

from multibatching.problem import Location, MultibatchingProblem, Product, TransportLink, TransportType
from multibatching.utils import generate_multibatching_problem

# MultibatchingProblem to JSON function
def export_multibatching_problem_to_json(problem: "MultibatchingProblem", file_path: str = None) -> str:
    """
    Converts a MultibatchingProblem object to a structured JSON string
    and optionally saves it to a file.
    """
    
    # 1. Serialize TransportTypes
    transport_types_data = []
    for tt in problem.transport_types:
        transport_types_data.append({
            "id": tt.id,
            "name": tt.name,
            "cost": tt.cost,
            "speed": tt.speed,
            "emissions": tt.emissions,
            "capacity": tt.capacity
        })

    # 2. Serialize Products
    products_data = []
    for p in problem.products:
        products_data.append({
            "id": p.id,
            "name": p.name,
            "size": p.size,
            "value": p.value,
            "valid_transport_ids": [t.id for t in p.valid_transports] # Convert the valid_transports set of objects to a list of IDs
        })

    # 3. Serialize Locations
    locations_data = []
    for loc in problem.locations:
        net_supply_json = {str(p.id): qty for p, qty in loc.net_supply.items()}
        
        locations_data.append({
            "id": loc.id,
            "name": loc.name,
            "net_supply": net_supply_json
        })

    # 4. Serialize TransportLinks
    links_data = []
    for link in problem.transport_links:
        links_data.append({
            "id": link.id,
            "location_l1_id": link.location_l1.id, # Replace Location and TransportType objects with their IDs (Foreign Keys)
            "location_l2_id": link.location_l2.id,
            "distance": link.distance,
            "transport_type_id": link.transport_type.id,
            "max_trips": link.max_trips
        })

    # 5. Final Structure
    json_structure = {
        "capital_factor": problem.capital_factor,
        "transport_types": transport_types_data,
        "products": products_data,
        "locations": locations_data,
        "transport_links": links_data
    }

    # Generate JSON string
    json_output = json.dumps(json_structure, indent=4)

    # Save to file if path is specified
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"JSON successfully exported to: {file_path}")

    return json_output

def load_multibatching_problem_from_json(json_input: Union[str, dict], verbose: bool = True) -> "MultibatchingProblem":
    """
    Reconstructs a MultibatchingProblem object from a JSON (string or dict).
    Performs referential integrity and logical feasibility checks.
    """
    
    # 1. Load raw data
    if isinstance(json_input, str):
        data = json.loads(json_input)
    else:
        data = json_input

    try:
        # Reconstruct TransportTypes
        transport_map = {}
        transport_types = []
        
        for tt_data in data.get("transport_types", []):
            tt = TransportType(
                id=tt_data["id"],
                name=tt_data.get("name"),
                cost=tt_data["cost"],
                speed=tt_data["speed"],
                emissions=tt_data["emissions"],
                capacity=tt_data["capacity"]
            )
            transport_types.append(tt)
            transport_map[tt.id] = tt

        # Reconstruct Products
        product_map = {}
        products = []
        
        for p_data in data.get("products", []):
            # Retrieve TransportType objects based on JSON IDs
            valid_transports_ids = p_data.get("valid_transport_ids", [])
            valid_transports = set()
            
            for tid in valid_transports_ids:
                if tid not in transport_map:
                    raise ValueError(f"Integrity Error: Product {p_data['id']} references TransportType ID {tid} which does not exist.")
                valid_transports.add(transport_map[tid])
            
            prod = Product(
                id=p_data["id"],
                name=p_data.get("name", ""),
                size=p_data["size"],
                value=p_data["value"],
                valid_transports=frozenset(valid_transports)
            )
            products.append(prod)
            product_map[prod.id] = prod

        # Reconstruct Locations 
        location_map = {}
        locations = []
        
        for l_data in data.get("locations", []):
            # Reconstruct net_supply dictionary (JSON keys are strings -> Product Objects)
            raw_supply = l_data.get("net_supply", {})
            net_supply = {}
            
            for pid_str, qty in raw_supply.items():
                pid = int(pid_str) # IDs in JSON keys are always strings
                if pid not in product_map:
                    raise ValueError(f"Integrity Error: Location {l_data['id']} references Product ID {pid} which does not exist.")
                net_supply[product_map[pid]] = qty
            
            loc = Location(
                id=l_data["id"],
                name=l_data.get("name", ""),
                net_supply=net_supply
            )
            locations.append(loc)
            location_map[loc.id] = loc

        # Reconstruct TransportLinks
        transport_links = []
        
        for link_data in data.get("transport_links", []):
            l1_id = link_data["location_l1_id"]
            l2_id = link_data["location_l2_id"]
            tt_id = link_data["transport_type_id"]
            
            # Referential integrity validations
            if l1_id not in location_map:
                raise ValueError(f"Link Error: Location L1 ID {l1_id} not found in link {link_data['id']}.")
            if l2_id not in location_map:
                raise ValueError(f"Link Error: Location L2 ID {l2_id} not found in link {link_data['id']}.")
            if tt_id not in transport_map:
                raise ValueError(f"Link Error: TransportType ID {tt_id} not found in link {link_data['id']}.")
            
            link = TransportLink(
                id=link_data["id"],
                location_l1=location_map[l1_id],
                location_l2=location_map[l2_id],
                distance=link_data["distance"],
                transport_type=transport_map[tt_id],
                max_trips=link_data.get("max_trips")
            )
            transport_links.append(link)

        # Create Problem Object
        problem = MultibatchingProblem(
            transport_types=transport_types,
            products=products,
            locations=locations,
            transport_links=transport_links,
            capital_factor=data.get("capital_factor", 0.1)
        )

        if verbose:
            print("Data structure reconstructed successfully.")
            print(f"- {len(products)} Products")
            print(f"- {len(locations)} Locations")
            print(f"- {len(transport_links)} Links")

        # Logical Verification (Feasibility Check)
        if verbose:
            print("\nVerifying logical feasibility (check_feasibility_per_product)...")
        
        is_feasible = problem.check_feasibility_per_product(verbose=verbose)
        
        if not is_feasible:
            print("\nWARNING: Problem loaded, but appears INFEASIBLE according to 'check_feasibility_per_product'.")
            print("This implies supply/demand mismatch or insufficient transport capacity.")
        else:
            if verbose: 
                print("\nSUCCESS: Problem is structurally valid and feasible per product.")

        return problem

    except KeyError as e:
        raise ValueError(f"JSON Format Error: Missing required field {e}")
    except Exception as e:
        raise ValueError(f"Error processing problem: {str(e)}")
    

def verify_equivalence(p1: "MultibatchingProblem", p2: "MultibatchingProblem") -> bool:
    """
    Manually verifies that two MultibatchingProblem objects are functionally identical.
    Raises AssertionError if discrepancies are found.
    """
    print("Starting deep equivalence verification...")

    # 1. Verify Global Factors
    assert math.isclose(p1.capital_factor, p2.capital_factor), \
        f"Different Capital Factor: {p1.capital_factor} vs {p2.capital_factor}"

    # Helpers to index by ID to facilitate comparisons
    def map_by_id(obj_list):
        return {obj.id: obj for obj in obj_list}

    t1_map = map_by_id(p1.transport_types)
    t2_map = map_by_id(p2.transport_types)
    prod1_map = map_by_id(p1.products)
    prod2_map = map_by_id(p2.products)
    loc1_map = map_by_id(p1.locations)
    loc2_map = map_by_id(p2.locations)
    link1_map = map_by_id(p1.transport_links)
    link2_map = map_by_id(p2.transport_links)

    # 2. Verify TransportTypes
    assert len(t1_map) == len(t2_map), "TransportTypes count differs"
    for tid, t1 in t1_map.items():
        t2 = t2_map.get(tid)
        assert t2 is not None, f"TransportType {tid} missing in loaded object"
        # Compare primitive fields
        assert t1.cost == t2.cost
        assert t1.speed == t2.speed
        assert t1.capacity == t2.capacity
        assert t1.emissions == t2.emissions
        assert t1.name == t2.name

    # 3. Verify Products
    assert len(prod1_map) == len(prod2_map), "Products count differs"
    for pid, pr1 in prod1_map.items():
        pr2 = prod2_map.get(pid)
        assert pr2 is not None, f"Product {pid} missing in loaded object"
        assert pr1.name == pr2.name
        assert pr1.size == pr2.size
        assert pr1.value == pr2.value
        
        # Compare valid_transports relation (comparing set IDs)
        vt1_ids = {t.id for t in pr1.valid_transports}
        vt2_ids = {t.id for t in pr2.valid_transports}
        assert vt1_ids == vt2_ids, f"Valid transports differ for Product {pid}"

    # 4. Verify Locations and Net Supply
    assert len(loc1_map) == len(loc2_map), "Locations count differs"
    for lid, l1 in loc1_map.items():
        l2 = loc2_map.get(lid)
        assert l2 is not None, f"Location {lid} missing in loaded object"
        assert l1.name == l2.name
        
        # Compare Net Supply (Dictionary Product -> int)
        # Convert keys (Product Objects) to their IDs for comparison
        supply1 = {p.id: qty for p, qty in l1.net_supply.items()}
        supply2 = {p.id: qty for p, qty in l2.net_supply.items()}
        
        # Verify supply dictionaries are identical
        assert supply1 == supply2, f"Net Supply differs in Location {lid}: {supply1} vs {supply2}"

    # 5. Verify TransportLinks
    assert len(link1_map) == len(link2_map), "TransportLinks count differs"
    for lid, link1 in link1_map.items():
        link2 = link2_map.get(lid)
        assert link2 is not None, f"Link {lid} missing in loaded object"
        assert link1.distance == link2.distance
        assert link1.max_trips == link2.max_trips
        
        # Verify referential integrity (IDs of connected objects)
        assert link1.location_l1.id == link2.location_l1.id
        assert link1.location_l2.id == link2.location_l2.id
        assert link1.transport_type.id == link2.transport_type.id

    print("SUCCESSFUL VERIFICATION: Loaded object is equivalent to the original.")
    return True

# EXAMPLE OF USE
if __name__ == "__main__":

    # Instanciate a random problem
    problem = generate_multibatching_problem(num_locations=60, num_transport_types=20, num_products=20)

    # Transform to JSON
    json_result = export_multibatching_problem_to_json(problem, "multibatching_problem.json")

    # Transform from JSON to problem
    problem_loaded = load_multibatching_problem_from_json(json_result)

    # Verify equivalence between original and reconstructed instances
    try:
        verify_equivalence(problem, problem_loaded)
    except AssertionError as e:
        print(f"VERIFICATION ERROR: {e}")

