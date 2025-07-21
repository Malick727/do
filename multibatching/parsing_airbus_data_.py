from multibatching.problem import Location, TransportType, Product, TransportLink
import json
import re
import os
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
                                            speed=speed_for_cp[idx],
                                            capacity=cap_for_cp[idx],
                                            emissions=co2_for_cp[idx]))
    return transportTypes, id_to_name


def parse_products(transport_types: list[TransportType],
                   id_to_name: dict[str, str]) -> list[Product]:
    tr_dict = {}
    transport_key_to_transport = {tr.id: tr for tr in transport_types}
    with open(os.path.join(data_folder, "products.json"), 'r') as filehandle:
        json_data = json.load(filehandle)
        names = [reformulate(x.get('name')) for x in json_data]
        sizes = [reformulate(x.get('length')) for x in json_data]
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
                start = line.find('demandSupply(')
                end = line.find(',')
                part = line[start + len('demandSupply('):end]
                location = re.findall(',(.*),', line)[0]
                nr = re.findall("\d+", line)[-1]
                if location not in location_dict:
                    location_dict[location] = {id_product_to_product[part]: int(nr)}
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


def main():
    transports, id_to_name = parse_transport_type()
    products = parse_products(transport_types=transports, id_to_name=id_to_name)
    locations, location_id_to_name = parse_locations(products=products)
    print(products)
    print([p.id for p in products])
    print(locations)
    transport_link = parse_routes(locations=locations,
                                  location_id_to_name=location_id_to_name,
                                  transport_type=transports,
                                  id_to_name_tr=id_to_name)
    print(transport_link)
    from multibatching.problem import MultibatchingProblem
    problem = MultibatchingProblem(transport_types=transports,
                                   products=products,
                                   locations=locations,
                                   transport_links=transport_link,
                                   capital_factor=0.1)
    print(problem)


if __name__ == "__main__":
    main()


