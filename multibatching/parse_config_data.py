import os
import json
import clingo
from dataclasses import dataclass, field
from typing import Hashable
from multibatching.problem import (MultibatchingProblem,
                                   TransportType,
                                   Product,
                                   Location,
                                   TransportLink)


# all_data_frames = {}
# # --- Parsing Function ---
# # globally defined variables
# # modelAggregateTemplates = [ {'name':'flowOut', 'filter':lambda s: s.name=='flowOut', 'columns':['part' ,'location', 'amount']},
# #                           {'name':'flowIn', 'filter':lambda s: s.name=='flowIn', 'columns':['part' ,'location', 'amount']},
# #                           {'name':'req', 'filter':lambda s: s.name=='req', 'columns':['part' ,'location', 'amount']},
# #                          {'name':'routeCostTotal', 'filter':lambda s: s.name=='routeCostTotal', 'columns':['from' ,'to', 'transport resource', 'costs']}
# #                           ]
# #
# # modelAtomTemplates = [
#  #     {'name':'route','filter':lambda s: s.name=='route', 'columns':['idx', 'from', 'to', 'tr', 'distance', 'cost']},
# # before
# # with ctl.solve(yield_=True, on_model=on_model) as hnd:
# #    for idx, mdl in enumerate(hnd):
# #       assignment_data = list(thy.assignment(mdl.thread_id))
#
# def parse_asp_model_to_solution(assignment_data, idx):
#     with ctl.solve(yield_=True, on_model=on_model) as hnd:
#         for idx, mdl in enumerate(hnd):
#            assignment_data = list(thy.assignment(mdl.thread_id))
#     for modelAggregateTemplate in modelAggregateTemplates:
#         dataName=modelAggregateTemplate["name"]
#         columns=modelAggregateTemplate["columns"]
#         filtered_data = [[*key.arguments, val] for key, val in assignment_data if key.name==dataName]
#         df = pd.DataFrame(data=filtered_data, columns=columns)
#         all_data_frames[dataName] = df
#
#
#
# def on_model(m: clingo.Model):
#         path_symbols = []
#         for name in [next(iter(d.values())) for d in modelAtomTemplates]:
#             path_symbols += list(filter(lambda s: s.name == name, m.symbols(atoms=True)))
#         for modelAtomTemplate in modelAtomTemplates:
#             filteredSymbols = list(filter(modelAtomTemplate['filter'], path_symbols))
#             arguments = list(map(lambda s: s.arguments, filteredSymbols))
#             df = pd.DataFrame(data=arguments, columns=modelAtomTemplate['columns'])
#             all_data_frames[modelAtomTemplate['name']] = df
#
#
#
#
# <<<<<<< HEAD
# def parse_problem_to_facts(mbp):
#     output_folder = os.path.join(this_folder, "lps/airbus/")
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     with open(os.path.join(output_folder, output_file), 'w') as file_out:
#
#
#         tp_name_asp =  "\n".join(f"transportationResource({tp.name})." for tp in mbp.transport_types)
#         tp_co2_asp = "\n".join(f"transportCO2({tp.name},{tp.emissions})." for tp in mbp.transport_types)
#         tp_cost_asp = "\n".join(f"transportCost({tp.name},{tp.cost})." for tp in mbp.transport_types)
#         tp_cap_asp =  "\n".join(f"transportCapacity({tp.name},{tp.capacity})." for tp in mbp.transport_types)
#         tp_speed_asp = "\n".join(f"transportSpeed({tp.name},{tp.speed})." for tp in mbp.transport_types)
#
#         transportationsString = tp_name_asp + tp_co2_asp + tp_cost_asp + tp_cap_asp + tp_speed_asp
#         file_out.writelines(transportationsString)
#
#         p_name_asp = "\n".join(f"part({product.name})." for product in mbp.products)
#         p_size_asp = "\n".join(f"part({product.name},{product.size})." for product in mbp.products)
#         p_validtp_asp = ""
#         for product in mbp.products:
#             p_validtp_asp += "\n".join(f"partTR({product.name},{validtp})." for validtp in products.valid_transports)
#
#         productString = p_name_asp + p_size_asp + p_validtp_asp
#         file_out.writelines(productString)
#
#         loc_name_asp = "\n".join(f"location({location.name})." for location in mbp.locations)
#         for location in mbp.locations:
#             net_supply_asp = "\n".join(f"demandOffer({location.name},{product},{val})." for product, val in location.net_supply.items())
#         locationString = loc_name_asp + net_supply_asp
#         file_out.writelines(locationString)
#
#         tp_cost_dict = {tp:tp.cost for tp in mbp.transport_types}
#         routes_asp =  "\n".join(f"route({tk.location_l1},{tk.location_l2},{tk.transport_type},{tk.distance},{compute_costs(tk.distance,tp_cost_dict.get(tp))})." for tk in mbp.transport_links)
#         file_out.writelines(routes_asp)
# =======
# # def parse_problem_to_facts(mbp):
# #     output_folder = os.path.join(this_folder, "lps/airbus/")
# #     if not os.path.exists(output_folder):
# #         os.makedirs(output_folder)
# #     with open(os.path.join(output_folder, output_file), 'w') as file_out:
# #
# #
# #         tp_name_asp =  "\n".join(f"transportationResource({tp.name})." for tp in mbp.transport_types)
# #         tp_co2_asp = "\n".join(f"transportCO2({tp.name},{tp.emissions})." for tp in mbp.transport_types)
# #         tp_cost_asp = "\n".join(f"transportCost({tp.name},{tp.cost})." for tp in mbp.transport_types)
# #         tp_cap_asp =  "\n".join(f"transportCapacity({tp.name},{tp.capacity})." for tp in mbp.transport_types)
# #         tp_speed_asp = "\n".join(f"transportSpeed({tp.name},{tp.speed})." for tp in mbp.transport_types)
# #
# #         transportationsString = tp_name_asp + tp_co2_asp + tp_cost_asp + tp_cap_asp + tp_speed_asp
# #         file_out.writelines(transportationsString)
# #
# #         p_name_asp = "\n".join(f"part({product.name})." for product in mbp.products)
# #         p_size_asp = "\n".join(f"part({product.name},{product.size})." for product in mbp.products)
# #         p_validtp_asp = ""
# #         for product in mbp.products:
# #             p_validtp_asp += "\n".join(f"partTR({product.name},{validtp})." for validtp in products.valid_transports)
# #
# #         productString = p_name_asp + p_size_asp + p_validtp_asp
# #         file_out.writelines(productString)
# #
# #         loc_name_asp = "\n".join(f"location({location.name})." for location in mbp.locations)
# #         for location in mbp.locations:
# #             net_supply_asp = "\n".join(f"demandOffer({location.name},{product},{val})." for product, val in location.net_supply.items())
# #         locationString = loc_name_asp + net_supply_asp
# #         file_out.writelines(locationString)
# #
# #         tp_cost_dict = {tp:tp.cost for tp in mbp.transport_types}
# #         routes_asp =  "\n".join(f"route({tk.location_l1},{tk.location_l2},{tk.transport_type},{tk.distance},{compute_costs(tk.distance,tp_cost_dict.get(tp))})." for for tk mbp.transport_links)
# #         file_out.writelines(routes_asp)
# >>>>>>> 991538f (minor fix in parse config)
#
#
# def compute_costs(distance,cost):
#     return distance*cost
#
#
#
#
#
#
#     json2asp(file_out)
#     output_file = "factsASP.lp"
#
#     MultibatchingProblem(
#         transport_types=list(transport_types_map.values()),
#         products=list(products_map.values()),
#         locations=locations_list,
#         transport_links=transport_links_list,
#         capital_factor=settings["capitalCosts"],
#         # co2_costs=settings["co2Costs"]
#     )
#
#
#
#
#
#



def parse_json_to_problem(json_data: dict, harbor_constraint = False,
                          scale_co2: float = 1) -> MultibatchingProblem:
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
            emissions=data["co2Emissions"]*scale_co2,
            capacity=data["capacity"]//10**4
        ) for _id, data in json_data["transportResources"].items()
    }

    # 2. Parse Products
    products_map = {
        _id: Product(
            id=_id,
            name=data["name"],
            # size=data["size"]//10**4,
            size=data["size"],
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
    # THis is not the good way to parse finally.
    # for strategy in json_data["sourcingStrategies"]:
    #    product = products_map[strategy["product"]]
    #    for loc_id, quantity in strategy["netSupplyDemand"].items():
    #        if loc_id in locations_map:
    #            locations_map[loc_id].net_supply[product] = quantity
    for product_key in json_data["products"]:
        product = products_map[product_key]
        for location_key in json_data["products"][product_key]["netSupplyDemand"]:
            locations_map[location_key].net_supply[product] = json_data["products"][product_key]["netSupplyDemand"][location_key]

    locations_list = list(locations_map.values())
    # 4. Parse Transport Links
    ships = ["Kugelbake", "Feeder Container Ship", "Truck General Cargo", "Oversized Tianjin RoRo Ship", "Truck on RoRo Ship", "Small Container Ship", "Airbus Mobile Express RoRo Ship", "Oversized River Transport", "Sea Large Container Ship", "Oversized Under Deck RoRo Ship" ]
    transport_links_list = []
    for _id, data in json_data["routes"].items():
        loc1 = locations_map[data["from"]]
        loc2 = locations_map[data["to"]]

        list_loc1_name = loc1.name.split(" ")
        is_harbor_1 = False
        if len(list_loc1_name)>1:
            last1 = list_loc1_name[-1]
            if last1 == "Harbor":
                is_harbor_1 = True

        list_loc2_name = loc2.name.split(" ")
        is_harbor_2 = False
        if len(list_loc2_name)>1:
            last2 = list_loc2_name[-1]
            if last1 == "Harbor":
                is_harbor_2 = True
        is_harbor = is_harbor_1 or is_harbor_2
        

        
        for tr_id, tr_data in data["transportResources"].items():
            transport_type = transport_types_map[tr_id]
            distance = tr_data["distance"]
            # if not(is_harbor) or (transport_type.name in ships) or not(harbor_constraint):
            transport_links_list.append(
                TransportLink(
                    # id = _id+"tr*id"+tr_id,
                    id = _id+"tr*id"+tr_id,
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


def parse_config_file_to_problem(json_path: str):
    dict_ = json.load(open(json_path, "r"))
    return parse_json_to_problem(dict_)


if __name__ == "__main__":
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "datalocal/config-local.json")
    import json
    dict_ = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(dict_)
    print(problem)
