import os
import json
from dataclasses import dataclass, field
from typing import Hashable
from multibatching.problem import (MultibatchingProblem,
                                   TransportType,
                                   Product,
                                   Location,
                                   TransportLink)


# --- Parsing Function ---

def parse_json_to_problem(json_data: dict) -> MultibatchingProblem:
    """
    Parses a JSON dictionary into a MultibatchingProblem object.
    """
    # 1. Parse Transport Types (from transportResources)
    transport_types_map = {
        _id: TransportType(
            id=_id,
            name=data["name"],
            cost=data["cost"],
            speed=data["speed"],
            emissions=data["co2Emissions"],
            capacity=1000
            # capacity=data["capacity"]//10**6
        ) for _id, data in json_data["transportResources"].items()
    }

    # 2. Parse Products
    products_map = {
        _id: Product(
            id=_id,
            name=data["name"],
            # size=data["size"]//10**6,
            size=100,
            value=data["value"],
            valid_transports=frozenset(
                transport_types_map[tr_id] for tr_id in data["validTR"]
            )
        ) for _id, data in json_data["products"].items()
    }

    # 3. Parse Locations
    locations_map = {
        _id: Location(id=_id,
                      name=data["name"],
                      net_supply={})
        for _id, data in json_data["locations"].items()
    }

    # Update locations with net supply/demand from sourcing strategies
    for strategy in json_data["sourcingStrategies"]:
        product = products_map[strategy["product"]]
        for loc_id, quantity in strategy["netSupplyDemand"].items():
            if loc_id in locations_map:
                locations_map[loc_id].net_supply[product] = quantity

    locations_list = list(locations_map.values())

    # 4. Parse Transport Links
    transport_links_list = []
    for _id, data in json_data["routes"].items():
        loc1 = locations_map[data["from"]]
        loc2 = locations_map[data["to"]]

        for tr_id, tr_data in data["transportResources"].items():
            transport_type = transport_types_map[tr_id]
            distance = tr_data["distance"]

            transport_links_list.append(
                TransportLink(
                    location_l1=loc1,
                    location_l2=loc2,
                    distance=distance,
                    transport_type=transport_type
                )
            )

    # 5. Extract settings
    settings = json_data["settings"]

    # 6. Create the final problem instance
    return MultibatchingProblem(
        transport_types=list(transport_types_map.values()),
        products=list(products_map.values()),
        locations=locations_list,
        transport_links=transport_links_list,
        capital_factor=settings["capitalCosts"],
        # co2_costs=settings["co2Costs"]
    )


if __name__ == "__main__":
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "datalocal/config-local.json")
    import json
    dict_ = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(dict_)
    print(problem)
    print("hey")
