import json
import os.path

location_name_id_mapping = {}
tr_name_id_mapping = {}
part_name_id_mapping = {}
tr_id_cost_mapping = {}

alternative_names = True
folder = '../../../multibatching/datalocal'
file_name = 'config-local' # 'oneheart-kerosene-aircraft' #'config-local.json' # 
input_file = f'{file_name}.json'

output_file = "factsASPalt.lp"

cap_size_divde = 1000000000

# only used for text and not 
def reformulate(text):

    temp_text = str(text)
    temp_text = temp_text.replace(".","")
    temp_text = temp_text.replace("-","")
    temp_text = temp_text.replace(" ","")
    return temp_text[0].lower() + temp_text[1:]

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
        tr_names = list(map(lambda x: f"transportResource({reformulate(transportation_data.get(x)['name'])}).\n", transportation_data))
        tr_co2 = list(map(lambda x: f"transportCO2({reformulate(transportation_data.get(x)['name'])},{max([1,int(reformulate(int(transportation_data.get(x)['co2Emissions'])))])}).\n",transportation_data))
        tr_costs = list(map(lambda x: f"transportCost({reformulate(transportation_data.get(x)['name'])},{max([1,int(reformulate(int(transportation_data.get(x)['cost'])))])}).\n", transportation_data))
        tr_capacities = list(map(lambda x: f"transportCapacity({reformulate(transportation_data.get(x)['name'])},{max([1,int(int(reformulate(int(transportation_data.get(x)['capacity'])))/cap_size_divde)])}).\n", transportation_data))
        tr_speeds = list(map(lambda x: f"transportSpeed({reformulate(transportation_data.get(x)['name'])},{max([1,int(reformulate(int(transportation_data.get(x)['speed'])))])}).\n", transportation_data))

        alt_tr_names_mapping = zip([reformulate(transportation_data.get(x)['name']) for x in transportation_data], [f"tr{n}" for n in range(0, len(transportation_data))]) 

        tr_names_for_mapping = list(map(lambda x: reformulate(transportation_data.get(x)['name']), transportation_data))        
        tr_ids_for_mapping = list(map(lambda x: x, transportation_data))
        tr_name_id_mapping = dict(zip(tr_ids_for_mapping,tr_names_for_mapping))
        tr_costs_for_mapping = list(map(lambda x: max([1,int(reformulate(int(transportation_data.get(x)['cost'])))]), transportation_data))
        tr_id_cost_mapping = dict(zip(tr_ids_for_mapping,tr_costs_for_mapping))

        routes_data = json_data["routes"]
        routes = ""

        for route in routes_data:
            route_dict = routes_data.get(route)
            source = location_name_id_mapping.get(route_dict["from"])
            destination = location_name_id_mapping.get(route_dict["to"])
            for tr_id in route_dict.get("transportResources"):
                tr_dict = route_dict.get("transportResources").get(tr_id)
                tr_name = tr_name_id_mapping.get(tr_id)
                distance = int(tr_dict["distance"])
                costs = int(tr_id_cost_mapping.get(tr_id) * distance)
                routes += (f"route({source},{destination},{tr_name},{distance},{costs}).\n")

    

        parts_data = json_data["products"]
        part_names = list(map(lambda x: f"part({reformulate(parts_data.get(x).get('name'))}).\n", parts_data))
        part_sizes = list(map(lambda x: f"partSize({reformulate(parts_data.get(x).get('name'))},{int(int(reformulate(parts_data.get(x).get('size')))/cap_size_divde)}).\n", parts_data))
        
        alt_part_names_mapping = zip([reformulate(parts_data.get(x)['name']) for x in parts_data], [f"part{n}" for n in range(0, len(parts_data))]) 


        parts_tr = ""
        for part in parts_data:
            part_dict = parts_data.get(part)
            name = reformulate(part_dict.get("name"))
            for tr in part_dict.get("validTR"):
                 parts_tr += f"partTR({name},{tr_name_id_mapping.get(str(tr))}).\n"

 
        #parts_names_for_mapping = list(map(lambda x: reformulate(parts_data.get(x).get('name')), parts_data))        
        #parts_ids_for_mapping = list(map(lambda x: x, parts_data))

        #part_name_id_mapping = dict(zip(parts_ids_for_mapping,parts_names_for_mapping))

        demandOffers = []
        for part in parts_data:
            part_dict = parts_data.get(part)
            part_name = reformulate(parts_data.get(part).get("name"))
            supply_demand_dict = part_dict.get("netSupplyDemand")
            demandOffers += list(map(lambda x: f"demandOffer({part_name},{location_name_id_mapping.get(x)},{supply_demand_dict.get(x)}).\n", supply_demand_dict))
            check_on_balance =  list(map(lambda x: supply_demand_dict.get(x), supply_demand_dict))
            #print("check_on_balance", part_name, sum(check_on_balance))
        
        text_to_be_written = "% ----------------------- locations -----------------------\n" + "".join(locations) \
                             + "\n% ----------------------- transport resources -----------------------\n" \
                             + "".join(tr_names) + "\n" +  "".join(tr_co2) + "\n" + "".join(tr_costs) + "\n" + "".join(tr_capacities) + "\n" + "".join(tr_speeds) + "\n\n" \
                             + "\n% ----------------------- routes -----------------------\n" + routes \
                             + "\n% ----------------------- parts -----------------------\n" \
                             + "".join(part_names) + "\n" +  "".join(part_sizes) + "\n" +  "".join(parts_tr) + "\n" \
                             + "\n% ----------------------- demands and offers -----------------------\n" + "".join(demandOffers) + "\n" \
                             + "% ----------------------- harbors -----------------------\n" \
                             + "harbor(naplesHarbor;marseilleHarbor;montoirHarbor;tunisHarbor;mobileHarbor;dunkerqueHarbor;gruenendeichHarbor;taichungHarbor;shanghaiHarbor;hamburgHarbor;gibraltarHarbor;portoHarbor;tianjinHarbor;casablancaHarbor;sacheonHarbor;stNazaire;houstonHarbor;savannaHarbor)."

        
        if alternative_names == True:
            # first replace harbor names to avoid wrong replacements
            for old_word, new_word in dict(alt_harbour_names_mapping).items():
                text_to_be_written = text_to_be_written.replace(old_word, new_word)
            alt_names_dict = dict(alt_prodloc_names_mapping) | dict(alt_tr_names_mapping) | dict(alt_part_names_mapping)
            for old_word, new_word in alt_names_dict.items():
                text_to_be_written = text_to_be_written.replace(old_word, new_word)
        file_out.writelines(text_to_be_written)



this_folder = os.path.abspath(os.path.dirname(__file__))
output_folder = os.path.join(this_folder, "lps/airbus/")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
with open(os.path.join(output_folder, output_file), 'w') as file_out:
    json2asp(file_out)
    print(f"successfully parsed ./{folder}/{input_file} and created {output_file} saved in {output_folder}.")