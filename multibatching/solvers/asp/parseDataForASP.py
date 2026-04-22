import json
import os.path

location_name_id_mapping = {}
tr_name_id_mapping = {}
product_name_id_mapping = {}
tr_id_cost_mapping = {}

alternative_names = True
folder = '../../../multibatching/datalocal'
file_name = 'config-local' # 'oneheart-kerosene-aircraft' #'config-local.json' # 
input_file = f'{file_name}.json'

output_file = "factsASP.lp"
cap_size_divde = 100000

# only used for text and not 
def reformulate(text):
    temp_text = str(text)
    temp_text = temp_text.replace(".","")
    temp_text = temp_text.replace("-","")
    temp_text = temp_text.replace(" ","")
    return temp_text[0].lower() + temp_text[1:]

def reformulateNr(text):
    temp_text = str(text)
    temp_text = temp_text.split(".")[0]
    temp_text = temp_text.replace("-","")
    temp_text = temp_text.replace(" ","")
    return int(temp_text[0].lower() + temp_text[1:])


def json2asp(file_out):

    with open(f"./{folder}/{input_file}", 'r') as filehandle:
        json_data = json.load(filehandle)
        locations_data = json_data["locations"]

        harbors = [reformulate(locations_data.get(loc)['name']) for loc in locations_data if locations_data.get(loc)['name'].endswith('Harbor')] 
        alt_harbour_names_mapping = zip(harbors,  [f"harbor{n}" for n in range(0, len(harbors))])

        production_locations =  [reformulate(locations_data.get(loc)['name']) for loc in locations_data if not loc.endswith('Harbor')] 
        alt_prodloc_names_mapping = zip(production_locations, [f"prodloc{n}" for n in range(0, len(production_locations))]) 
        
        locations = list(map(lambda x: f"location({reformulate(locations_data.get(x)['name'])}).\n", locations_data))

        loc_names_for_mapping = list(map(lambda x: f"{reformulate(locations_data.get(x)['name'])}", locations_data))    
        loc_ids_for_mapping =  list(map(lambda x: str(x), locations_data))
        location_name_id_mapping = dict(zip(loc_ids_for_mapping,loc_names_for_mapping))
        
        transportation_data = json_data["transportResources"]
        tr_capacities = list(map(lambda x: f"transportCapacity({reformulate(transportation_data.get(x)['name'])},{max([1,reformulateNr(int(transportation_data.get(x)['capacity'])/cap_size_divde)])}).\n", transportation_data))
        
        tr_to_delete = []    
        for tr_cap in tr_capacities:
            if tr_cap.endswith("1).\n"):
                end_loc = tr_cap.find(',')
                tr_name = tr_cap[18:end_loc]
                tr_to_delete.append(tr_name)
        print(tr_to_delete)

        transportation_data_filter = transportation_data #filter(lambda x: reformulate(transportation_data.get(x)['name']) not in tr_to_delete, transportation_data)
        tr_capacities = list(map(lambda x: f"transportCapacity({reformulate(transportation_data.get(x)['name'])},{max([1,reformulateNr(int(transportation_data.get(x)['capacity'])/cap_size_divde)])}).\n",transportation_data_filter))

        tr_names = list(map(lambda x: f"transportResource({reformulate(transportation_data.get(x)['name'])}).\n", transportation_data_filter))
        tr_co2 = list(map(lambda x: f"transportCO2({reformulate(transportation_data.get(x)['name'])},{max([1,reformulateNr(transportation_data.get(x)['co2Emissions'])])}).\n", transportation_data_filter))
        tr_costs = list(map(lambda x: f"transportCost({reformulate(transportation_data.get(x)['name'])},{max([0,reformulateNr(transportation_data.get(x)['cost'])])}).\n", transportation_data_filter))
        tr_speeds = list(map(lambda x: f"transportSpeed({reformulate(transportation_data.get(x)['name'])},{max([1,reformulateNr(transportation_data.get(x)['speed'])])}).\n",  transportation_data_filter))
        tr_costPerCap = list(map(lambda x: f"transportCostPerCap({reformulate(transportation_data.get(x)['name'])},{max([1,reformulateNr(int((transportation_data.get(x)['capacity']/cap_size_divde)/transportation_data.get(x)['cost']))])}).\n",  transportation_data_filter))        
        

        alt_tr_names_mapping = zip([reformulate(transportation_data.get(x)['name']) for x in transportation_data], [f"tr{n}" for n in range(0, len(transportation_data))]) 

        tr_names_for_mapping = list(map(lambda x: reformulate(transportation_data.get(x)['name']), transportation_data))        
        tr_ids_for_mapping = list(map(lambda x: x, transportation_data))
        tr_name_id_mapping = dict(zip(tr_ids_for_mapping,tr_names_for_mapping))
        tr_costs_for_mapping = list(map(lambda x: max([1,reformulateNr(transportation_data.get(x)['cost'])]), transportation_data))
        tr_id_cost_mapping = dict(zip(tr_ids_for_mapping,tr_costs_for_mapping))

        routes_data = json_data["routes"]
        routes = ""

        idx = 0
        for route in routes_data:
            route_dict = routes_data.get(route)
            source = location_name_id_mapping.get(route_dict["from"])
            destination = location_name_id_mapping.get(route_dict["to"])
            for tr_id in route_dict.get("transportResources"):
                tr_dict = route_dict.get("transportResources").get(tr_id)
                tr_name = tr_name_id_mapping.get(tr_id)
                distance = int(tr_dict["distance"])
                costs = int(tr_id_cost_mapping.get(tr_id) * distance)
                routes += (f"route({idx},{source},{destination},{tr_name},{distance},{costs}).\n")
                idx += 1

        products_data = json_data["products"]
        product_names = list(map(lambda x: f"product({reformulate(products_data.get(x).get('name'))}).\n", products_data))
        product_sizes = list(map(lambda x: f"productSize({reformulate(products_data.get(x).get('name'))},{reformulateNr(products_data.get(x).get('size')/cap_size_divde)}).\n", products_data))
        
        alt_product_names_mapping = zip([reformulate(products_data.get(x)['name']) for x in products_data], [f"product{n}" for n in range(0, len(products_data))]) 

        products_tr = ""
        for product in products_data:
            product_dict = products_data.get(product)
            name = reformulate(product_dict.get("name"))
            size = products_data.get(product).get('size')
            for tr in product_dict.get("validTR"):
                #if int(transportation_data.get(tr)['capacity']) > int(size):
                products_tr += f"productTR({name},{tr_name_id_mapping.get(str(tr))}).\n"
                #else:
                #    print(tr_name_id_mapping.get(str(tr)), name)
                #    print(f'{int(transportation_data.get(tr)['capacity'])} < {int(size)}')
 
        #products_names_for_mapping = list(map(lambda x: reformulate(products_data.get(x).get('name')), products_data))        
        #products_ids_for_mapping = list(map(lambda x: x, products_data))

        #product_name_id_mapping = dict(zip(products_ids_for_mapping,products_names_for_mapping))

        demandOffers = []
        for product in products_data:
            product_dict = products_data.get(product)
            product_name = reformulate(products_data.get(product).get("name"))
            supply_demand_dict = product_dict.get("netSupplyDemand")
            demandOffers += list(map(lambda x: f"demandOffer({product_name},{location_name_id_mapping.get(x)},{supply_demand_dict.get(x)}).\n", supply_demand_dict))
            
        text_to_be_written = "% ----------------------- locations -----------------------\n" + "".join(locations) \
                             + "\n% ----------------------- transport resources -----------------------\n" \
                             + "".join(tr_names) + "\n" +  "".join(tr_co2) + "\n" + "".join(tr_costs) + "\n" + "".join(tr_capacities) + "\n" \
                              + "".join(tr_speeds) + "\n"  +  "".join(tr_costPerCap) + "\n\n" \
                             + "\n% ----------------------- routes -----------------------\n" + routes \
                             + "\n% ----------------------- products -----------------------\n" \
                             + "".join(product_names) + "\n" +  "".join(product_sizes) + "\n" +  "".join(products_tr) + "\n" \
                             + "\n% ----------------------- demands and offers -----------------------\n" + "".join(demandOffers) + "\n" \
                             + "% ----------------------- harbors -----------------------\n" \
                             + "harbor(naplesHarbor;marseilleHarbor;montoirHarbor;tunisHarbor;mobileHarbor;dunkerqueHarbor;gruenendeichHarbor;taichungHarbor;shanghaiHarbor;hamburgHarbor;gibraltarHarbor;portoHarbor;tianjinHarbor;casablancaHarbor;sacheonHarbor;stNazaire;houstonHarbor;savannaHarbor)." + "\n" \
                             + "ship(kugelbake;feederContainerShip;truckGeneralCargo;oversizedTianjinRoRoShip;truckonRoRoShip;smallContainerShip;airbusMobileExpressRoRoShip;oversizedRiverTransport;seaLargeContainerShip;oversizedUnderDeckRoRoShip)."

        
        if alternative_names == True:
            # first replace harbor names to avoid wrong replacements
            for old_word, new_word in dict(alt_harbour_names_mapping).items():
                text_to_be_written = text_to_be_written.replace(old_word, new_word)
            alt_names_dict = dict(alt_prodloc_names_mapping) | dict(alt_tr_names_mapping) | dict(alt_product_names_mapping)
            for old_word, new_word in alt_names_dict.items():
                text_to_be_written = text_to_be_written.replace(old_word, new_word)
        file_out.writelines(text_to_be_written)




this_folder = os.path.abspath(os.path.dirname(__file__))
output_folder = os.path.join(this_folder, "")
with open(os.path.join(output_folder, "req.lp"), 'w') as file_out:
    lines = ""
#    
    with open(f"/Users/thd7as/Documents/GitHub/discrete-optimisation-use-cases/multibatching/solvers/asp/requiredNet.csv", 'r') as filehandle:
        for flow in filehandle:
            lines += f"requiredNet({flow.strip()}).\n"
    file_out.write(lines)

#this_folder = os.path.abspath(os.path.dirname(__file__))
#output_folder = os.path.join(this_folder, "lps/airbus/")
#if not os.path.exists(output_folder):
#    os.makedirs(output_folder)
#with open(os.path.join(output_folder, output_file), 'w') as file_out:
#    json2asp(file_out)
#    print(f"successfully parsed ./{folder}/{input_file} and created {output_file} saved in {output_folder}.")

#this_folder = os.path.abspath(os.path.dirname(__file__))
#output_folder = os.path.join(this_folder, "lps/airbus/")
#with open(os.path.join(output_folder, "possible_flows_new.lp"), 'w') as file_out:
#    lines = ""
#    
#    with open(f"/Users/thd7as/Documents/GitHub/discrete-optimisation-use-cases/multibatching/solvers/asp/lps/airbus/possibleFlows.lp", 'r') as filehandle:
#        for flow in filehandle:
#            lines += f"possibleFlow({flow.strip()}).\n"
#    file_out.write(lines)
#def mapToAsp(predicateConstantsDict):
#    aspString = "".join(f"{predicate}({','.join(str(n) for n in constants)}).\n" for predicate, constants in predicateConstantsDict.items())
#    return aspString

