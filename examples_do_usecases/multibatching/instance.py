from multibatching.problem import *


def init_instance():
    # --- 1. Define TransportType Instances ---
    tr1 = TransportType(id=1, cost=56, speed=0.3, emissions=60, capacity=10)
    tr2 = TransportType(id=2, cost=42.5, speed=0.2, emissions=45, capacity=15)

    transport_types_list = [tr1, tr2]

    # --- 2. Define Product Instances ---
    # Valid transports are frozensets of TransportType objects
    p1 = Product(id=1, name="p1", size=4, value=1000, valid_transports=frozenset([tr1, tr2]))
    p2 = Product(id=2, name="p2", size=3, value=600, valid_transports=frozenset([tr1, tr2]))

    products_list = [p1, p2]

    # --- 3. Define Location Instances with Net Supply (Scenario 2 data) ---
    # Initialize net_supply for each location with zeros for all products
    # Then update based on scenario 2 demands
    all_products_dict = {p.id: p for p in products_list}

    l1_net_supply = {all_products_dict[1]: 3, all_products_dict[2]: 10}
    l2_net_supply = {all_products_dict[1]: 10, all_products_dict[2]: -5}
    l3_net_supply = {} # No direct supply/demand in thesis examples, but is a node
    l4_net_supply = {} # No direct supply/demand
    l5_net_supply = {all_products_dict[1]: -10, all_products_dict[2]: -2}
    l6_net_supply = {all_products_dict[1]: -3, all_products_dict[2]: -3} # p2: -10 for l6

    l1 = Location(id='l1', net_supply=l1_net_supply)
    l2 = Location(id='l2', net_supply=l2_net_supply)
    l3 = Location(id='l3', net_supply=l3_net_supply)
    l4 = Location(id='l4', net_supply=l4_net_supply)
    l5 = Location(id='l5', net_supply=l5_net_supply)
    l6 = Location(id='l6', net_supply=l6_net_supply)

    locations_list = [l1, l2, l3, l4, l5, l6]

    # Helper to map location IDs to Location objects
    loc_map = {loc.id: loc for loc in locations_list}

    # --- 4. Define TransportLink Instances (based on routes_info from CP-SAT script) ---
    transport_links_list = [
        TransportLink(location_l1=loc_map['l1'], location_l2=loc_map['l3'], distance=1, transport_type=tr1),  # r2
        # TransportLink(location_l1=loc_map['l1'], location_l2=loc_map['l3'], distance=3, transport_type=tr1), # r3
        TransportLink(location_l1=loc_map['l1'], location_l2=loc_map['l3'], distance=4, transport_type=tr2),  # r4
        TransportLink(location_l1=loc_map['l1'], location_l2=loc_map['l4'], distance=4, transport_type=tr1),  # r5
        TransportLink(location_l1=loc_map['l1'], location_l2=loc_map['l2'], distance=2, transport_type=tr2),  # r8
        TransportLink(location_l1=loc_map['l2'], location_l2=loc_map['l4'], distance=2, transport_type=tr1),  # r9
        TransportLink(location_l1=loc_map['l2'], location_l2=loc_map['l4'], distance=6, transport_type=tr2),  # r10
        TransportLink(location_l1=loc_map['l2'], location_l2=loc_map['l6'], distance=1.5, transport_type=tr1),  # r11
        TransportLink(location_l1=loc_map['l4'], location_l2=loc_map['l6'], distance=5, transport_type=tr2),  # r12
        TransportLink(location_l1=loc_map['l4'], location_l2=loc_map['l6'], distance=3, transport_type=tr1),  # r13
        TransportLink(location_l1=loc_map['l4'], location_l2=loc_map['l5'], distance=4, transport_type=tr2),  # r14
        TransportLink(location_l1=loc_map['l4'], location_l2=loc_map['l5'], distance=2, transport_type=tr1),  # r15
        TransportLink(location_l1=loc_map['l5'], location_l2=loc_map['l3'], distance=3, transport_type=tr2),  # r16
        TransportLink(location_l1=loc_map['l3'], location_l2=loc_map['l5'], distance=1, transport_type=tr1),  # r17
    ]

    

    # --- 5. Create MultibatchingProblem Instance ---
    problem_instance = MultibatchingProblem(
        transport_types=transport_types_list,
        products=products_list,
        locations=locations_list,
        transport_links=transport_links_list
    )

    print("MultibatchingProblem instance created successfully!")
    print(f"Number of transport types: {problem_instance.nb_transport_types}")
    print(f"Number of products: {problem_instance.nb_products}")
    print(f"Number of locations: {problem_instance.nb_locations}")
    print(f"Number of transport links: {problem_instance.nb_transport_links}")

    print("\n--- Example Access (l1 net supply for p1 and p2) ---")
    # Find p1 and p2 objects to access net_supply correctly
    p1_obj = next(p for p in products_list if p.id == 1)
    p2_obj = next(p for p in products_list if p.id == 2)

    l1_obj = next(loc for loc in locations_list if loc.id == 'l1')
    print(f"Net supply for {p1_obj.id} at {l1_obj.id}: {l1_obj.net_supply.get(p1_obj, 0)}")
    print(f"Net supply for {p2_obj.id} at {l1_obj.id}: {l1_obj.net_supply.get(p2_obj, 0)}")

    print("\n--- Example Access (a transport link) ---")
    first_link = transport_links_list[0]
    print(f"First transport link: {first_link.location_l1.id} -> {first_link.location_l2.id}, "
          f"distance={first_link.distance}, transport_type={first_link.transport_type.id}")
    return problem_instance