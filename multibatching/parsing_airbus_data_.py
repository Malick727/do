from multibatching.problem import Location, TransportType, Product, TransportLink
import json
import re
import os

from multibatching.utils import basic_sanity_check

this_folder = os.path.abspath(os.path.dirname(__file__))
data_folder = os.path.join(this_folder, "datalocal/")
location_name_id_mapping = {}
tr_name_id_mapping = {}
part_name_id_mapping = {}
tr_id_cost_mapping = {}


def reformulate(text):
    temp_text = str(text)
    temp_text = temp_text.replace(".", "")
    temp_text = temp_text.replace("-", "")
    temp_text = temp_text.replace(" ", "")
    return temp_text[0].lower() + temp_text[1:]


# def write_tr_infos(file_out):
def parse_transport_type() -> tuple[list[TransportType], dict[str, str]]:
    with open(os.path.join(data_folder,
                           "transportResources.json"), 'r') as filehandle:
        json_data = json.load(filehandle)
        name_for_cp = [reformulate(x.get('name')) for x in json_data]
        cost_for_cp = [float(x.get('recurringCost')) for x in json_data]
        speed_for_cp = [x.get('speed') for x in json_data]
        co2_for_cp = [float(x.get('co2Emissions')) for x in json_data]
        cap_for_cp = [x.get('capacity') for x in json_data]

        names_for_mapping = list(map(lambda x: reformulate(x.get('name')), json_data))
        ids_for_mapping = list(map(lambda x: x.get('id'), json_data))
        id_to_name = dict(zip(ids_for_mapping, names_for_mapping))
    transportTypes = []
    for idx, name in enumerate(name_for_cp):
        transportTypes.append(TransportType(id=name,
                                            cost=int(cost_for_cp[idx]),
                                            speed=10000,  # speed_for_cp[idx],
                                            capacity=10000000000,  # cap_for_cp[idx],
                                            emissions=co2_for_cp[idx]))
    return transportTypes, id_to_name


def parse_products(transport_types: list[TransportType],
                   id_to_name: dict[str, str]) -> list[Product]:
    tr_dict = {}
    transport_key_to_transport = {tr.id: tr for tr in transport_types}
    with open(os.path.join(data_folder, "products.json"), 'r') as filehandle:
        json_data = json.load(filehandle)
        names = [reformulate(x.get('name')) for x in json_data]
        sizes = [float(x.get('length')) for x in json_data]
        for item in json_data:
            name = reformulate(item.get("name"))
            tr_list = []
            for tr in item.get("transportResources"):
                print(transport_key_to_transport)
                tr_list.append(transport_key_to_transport[id_to_name[tr]])
            tr_dict[name] = frozenset(tr_list)
    productTypes = []
    for idx, name in enumerate(names):
        productTypes.append(Product(id=name, size=sizes[idx], value=0,
                                    valid_transports=tr_dict.get(name)))
    return productTypes


def parse_locations(products: list[Product]) -> tuple[list[Location], dict[str, str]]:
    id_product_to_product = {p.id: p for p in products}

    with open(os.path.join(data_folder, "demand_supply.lp"), 'r') as filehandle:
        location_dict = {}
        for line in filehandle:
            if line.startswith("demandSupply"):
                content_in_parentheses = line[len("demandSupply("):-2]
                parts = content_in_parentheses.split(',')
                part = parts[0]
                location = parts[1]
                nr = int(parts[2].replace(")", ""))
                if location not in location_dict:
                    location_dict[location] = {id_product_to_product[part]: int(nr)}
                else:
                    if id_product_to_product[part] in location_dict[location]:
                        print("already there")
                        location_dict[location][id_product_to_product[part]] += int(nr)
                    else:
                        location_dict[location].update({id_product_to_product[part]: int(nr)})
    with open(os.path.join(data_folder, "locations.json"), 'r') as filehandle:
        json_data = json.load(filehandle)
        location_names = [reformulate(x.get('name')) for x in json_data]
        location_id_to_name = {x["id"]: reformulate(x.get('name')) for x in json_data}
        locations_list = []
        for idx, name in enumerate(location_names):
            if name in location_dict:
                locations_list.append(Location(id=name, net_supply=location_dict[name]))
            else:
                locations_list.append(Location(id=name, net_supply={}))
    return locations_list, location_id_to_name


def parse_routes(locations: list[Location],
                 location_id_to_name: dict[str, str],
                 transport_type: list[TransportType],
                 id_to_name_tr: dict[str, str]) -> list[TransportLink]:
    location_name_to_location = {locations[i].id: locations[i] for i in range(len(locations))}
    transport_id_to_transport = {transport_type[i].id: transport_type[i] for i in range(len(transport_type))}
    with open(os.path.join(data_folder, "transportRoutes.json"), 'r') as filehandle:
        json_data = json.load(filehandle)
        transport_links_list = []
        for route in json_data:
            source = location_id_to_name[route.get("sourceId")]
            destination = location_id_to_name[route.get("destinationId")]
            print(source, destination)
            for tr in route.get("transportResources"):
                tr_id = tr.get("transportResourceId")
                tr_name = id_to_name_tr[tr_id]
                distance = int(tr.get("distance"))
                transport_links_list.append(TransportLink(location_l1=location_name_to_location[source],
                                                          location_l2=location_name_to_location[destination],
                                                          distance=distance,
                                                          transport_type=transport_id_to_transport[tr_name]))
    return transport_links_list


def main_parse():
    transports, id_to_name = parse_transport_type()
    products = parse_products(transport_types=transports, id_to_name=id_to_name)
    locations, location_id_to_name = parse_locations(products=products)
    transport_link = parse_routes(locations=locations,
                                  location_id_to_name=location_id_to_name,
                                  transport_type=transports,
                                  id_to_name_tr=id_to_name)
    # locations = [l for l in locations if l.net_supply != {}]
    # transport_link = [tl for tl in transport_link if tl.location_l1 in locations and tl.location_l2 in locations]
    from multibatching.problem import MultibatchingProblem
    problem = MultibatchingProblem(transport_types=transports,
                                   products=products,
                                   locations=locations,
                                   transport_links=transport_link,
                                   capital_factor=0.1)
    for l in problem.locations:
        print(l.id, ", :")
        for p in l.net_supply:
            print(p.id, ": ", l.net_supply[p])
        print("----")
    total_flow = basic_sanity_check(problem)
    ps = [p for p in problem.products if total_flow[p] == 0 and p.id != "section2"]
    locations_ = []
    for l in locations:
        locations_.append(Location(id=l.id,
                                   net_supply={p: l.net_supply[p] for p in l.net_supply if p in ps}))
    problem.products = ps
    problem = MultibatchingProblem(transport_types=transports,
                                   products=ps,
                                   locations=locations_,
                                   transport_links=transport_link)
    problem.check_feasibility_per_product(verbose=True)
    return problem


if __name__ == "__main__":
    main_parse()


