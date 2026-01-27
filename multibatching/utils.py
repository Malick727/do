from collections import defaultdict

from multibatching.problem import MultibatchingProblem, MultibatchingSolution
from multibatching.problem import *
import networkx as nx
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
from collections import defaultdict
from multibatching.problem import MultibatchingProblem, MultibatchingSolution, Product
from typing import List, Optional
import numpy as np
import os
import json
import pandas as pd
import time
from plotly.express.colors import qualitative
try:
    from geopy.geocoders import Nominatim
    from pyproj import Geod
    import searoute as sr
except:
    pass


def plot_solution_graph(solution: MultibatchingSolution, problem: MultibatchingProblem):
    """
    Renders the product flows of a MultibatchingSolution on a graph.

    Args:
        solution: An instance of MultibatchingSolution containing the list of packing flows.
        problem: An instance of MultibatchingProblem containing the problem definition
                 (locations, transport links, products).
    """

    G = nx.DiGraph()  # Create a directed graph

    # --- 1. Add Nodes (Locations) with Net Supply Information ---
    node_labels = {}
    for loc in problem.locations:
        G.add_node(loc.id, type='location', obj=loc)

        # Format net_supply for node labels
        supply_details = []
        # Sort products by ID for consistent display
        sorted_products = sorted(problem.products, key=lambda p: p.id)
        for product in sorted_products:
            supply_qty = loc.net_supply.get(product, 0)
            if supply_qty != 0:
                # Use a plus sign for supply, minus for demand, for clarity
                sign = "+" if supply_qty > 0 else ""
                supply_details.append(f"P{product.id}: {sign}{supply_qty}")

        if supply_details:
            node_labels[loc.id] = f"{loc.name}\n({', '.join(supply_details)})"
        else:
            node_labels[loc.id] = f"{loc.name}"

    # --- 2. Aggregate Product Flows per (from_loc, to_loc) pair and collect packing info ---
    # Dictionary to store aggregated flow for each (from_loc, to_loc, product)
    aggregated_product_flows_per_pair = defaultdict(lambda: defaultdict(float))

    # Store total flow for each (from_loc, to_loc) pair to determine edge thickness
    total_flow_per_pair = defaultdict(float)

    # Store individual PackingTransport details per (from_loc, to_loc) pair for detailed labels
    # Key: (from_loc_id, to_loc_id) -> Value: List of formatted packing strings
    detailed_packing_labels_per_pair = defaultdict(list)

    # Map product IDs to colors for clearer distinction in labels
    product_colors_map = {
        1: 'blue',
        2: 'darkorange',
        3: 'green',
        4: 'red',
        5: 'purple',
        6: 'brown',
        # Add more colors if more products are possible
    }

    for flow_item in solution.list_flows:
        from_loc_id = flow_item.transport_link.location_l1.id
        to_loc_id = flow_item.transport_link.location_l2.id
        current_pair = (from_loc_id, to_loc_id)

        # Aggregate total product flow for edge width
        for product, units in flow_item.product_packing.items():
            if product in problem.products:  # Ensure product is part of the problem's defined products
                aggregated_product_flows_per_pair[current_pair][product] += units * flow_item.nb_packing
                total_flow_per_pair[current_pair] += units * flow_item.nb_packing

                # Prepare the detailed packing string for this flow_item
        packing_details_str = []
        # Sort products in packing for consistent label order
        sorted_packing_products = sorted(flow_item.product_packing.keys(), key=lambda p: p.id)
        for product in sorted_packing_products:
            units = flow_item.product_packing[product]
            if product in problem.products:
                packing_details_str.append(f"{units}xP{product.name}")

        # Ensure that nb_packing is displayed as a float for consistency with problem definition
        nb_packing_display = f"{flow_item.nb_packing:.1f}".rstrip('0').rstrip('.')  # Remove trailing .0 if integer

        detailed_label_line = (
            f"[{', '.join(packing_details_str)}] "
            f"@ {nb_packing_display} "
            f"(TR: {flow_item.transport_link.transport_type.name})"
        )
        detailed_packing_labels_per_pair[current_pair].append(detailed_label_line)

    # --- 3. Add Edges (TransportLinks) with Flow Information ---
    edge_label_data = {}
    edge_widths_dict = {}

    # Calculate max_flow from the new total_flow_per_pair
    max_flow = max(total_flow_per_pair.values()) if total_flow_per_pair else 1

    # Iterate through unique (from_loc, to_loc) pairs that have flows
    for current_pair in total_flow_per_pair:
        from_node = current_pair[0]
        to_node = current_pair[1]

        current_pair_total_flow = total_flow_per_pair[current_pair]

        if current_pair_total_flow > 0:  # Only draw edges with actual flow
            G.add_edge(from_node, to_node,
                       total_flow=current_pair_total_flow)

            # Use the pre-prepared detailed_packing_labels_per_pair for this edge
            edge_label_data[current_pair] = detailed_packing_labels_per_pair[current_pair]

            # Scale edge width based on total flow for this pair
            width = 0.5 + 5 * (current_pair_total_flow / max_flow)
            edge_widths_dict[current_pair] = width

    if not G.edges():
        print("Aucun flux de produit actif à afficher.")
        plt.figure(figsize=(8, 6))
        plt.title("Réseau logistique (Aucun flux à afficher)")
        nx.draw_networkx(G, with_labels=True, node_color='lightgray', node_size=1000)
        plt.show()
        return

    # --- 4. Plotting ---
    # Adjust figure size dynamically based on number of nodes for better scaling
    fig_width = max(10, problem.nb_locations * 2)
    fig_height = max(8, problem.nb_locations * 1.5)
    plt.figure(figsize=(fig_width, fig_height))

    # Use a dynamic layout for better visualization
    # Spring layout is generally good for arbitrary graphs
    pos = nx.spring_layout(G, k=0.8 / G.number_of_nodes() ** 0.5, iterations=50,
                           # seed=random.randint(0, 10000)
                           )
    # Use circular_layout for smaller, dense graphs if spring_layout is too messy
    pos = nx.circular_layout(G)

    # Draw nodes
    # Adjust node size based on number of locations
    node_size_base = 2500
    node_size_adjusted = node_size_base * (6 / problem.nb_locations) if problem.nb_locations > 6 else node_size_base
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=node_size_adjusted, alpha=0.9, linewidths=1.5,
                           edgecolors='gray')

    # Draw node labels including net supply
    # Adjust font size based on number of locations
    font_size_nodes = max(7, 10 - problem.nb_locations // 2)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size_nodes, font_weight='bold')

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=[e for e in G.edges()],
                           width=[edge_widths_dict[e] for e in G.edges()],
                           arrowstyle='->', arrowsize=20,
                           edge_color='dimgray', alpha=0.7)

    # Draw edge labels (detailed packing, frequency, TR)
    for (u, v), list_of_detailed_labels in edge_label_data.items():
        # Get midpoint of the edge
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2

        # Calculate perpendicular offset for labels
        dx = pos[v][0] - pos[u][0]
        dy = pos[v][1] - pos[u][1]
        norm = (dx ** 2 + dy ** 2) ** 0.5

        # Adjust offset magnitude based on graph density/size for better visibility
        offset_magnitude = 0.05 + 0.005 * problem.nb_locations
        offset_x_perp = -dy / norm * offset_magnitude if norm > 0 else 0
        offset_y_perp = dx / norm * offset_magnitude if norm > 0 else 0

        # Adjust line height and font size for multiple lines of text on edges
        line_height_factor = 0.04 * (6 / problem.nb_locations) if problem.nb_locations > 6 else 0.04
        font_size_edges = max(6, 7 - problem.nb_locations // 3)
        total_text_height = len(list_of_detailed_labels) * line_height_factor
        current_y_offset = y + offset_y_perp + (total_text_height / 2) - (line_height_factor / 2)

        for label_text in list_of_detailed_labels:
            plt.text(x + offset_x_perp, current_y_offset, label_text,
                     horizontalalignment='center', verticalalignment='center',
                     color='black',  # Default color for detailed packing labels
                     fontweight='normal',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'),
                     fontsize=font_size_edges)

            current_y_offset -= line_height_factor

    plt.title("Flux de produits détaillés dans le réseau logistique mondial", size=15)
    plt.axis('off')  # Turn off the axis
    plt.tight_layout()
    plt.show()


def get_coords(city_name: str) -> Optional[tuple[float, float]]:
    """Finds latitude and longitude for a city and caches the result."""
    geolocator = Nominatim(user_agent="city_coordinate_finder_v1")
    try:
        if "Harbor" in city_name:
            city_name = city_name.split(" ")[0]
        elif "Ville de" in city_name:
            city_name = city_name.split("Ville de")[1]
        print(f"Geocoding {city_name}...")
        location = geolocator.geocode(city_name)
        time.sleep(1) # Be respectful of Nominatim's usage policy
        if location:
            print(location)
            coords = (location.latitude, location.longitude)
            return coords
    except Exception as e:
        print(f"Error geocoding {city_name}: {e}")
    return None


def getFALsLocation(problem):
    df = pd.DataFrame(columns=["lat","lon"])

    for city in ["Mobile","Toulouse","Hambourg","Tianjin"]:
        (lat,lon) = get_coords(city)
        df.loc[city] = [lat, lon]
    return df

def formatCoordinates():
    with open("multibatching/datalocal/coordinates.json", "r") as f :
        coordinates = json.load(f)
    df = pd.DataFrame(columns=["lat","lon", "type"])
    fals = ["Mobile","Toulouse","Hamburg","Tianjin"]
    for city, val in coordinates.items():
        if city in fals:
            type_factory = "FAL"
        else:
            type_factory = "MCA"
        df.loc[city] = [val[0], val[1], type_factory]
    return df




def plot_solution_on_map(solution: MultibatchingSolution):
    """
    Creates an interactive Plotly map of a MultibatchingSolution.
    """
    # 1. Extract data from the solution object
    
    locations_data = {}
    

    # coordinates = {}
    # for l in solution.problem.locations:
    #     coordinates[l.name] = get_coords(l.name)
    

    # with open("multibatching/datalocal/coordinates.json", "w") as f :
    #     json.dump(coordinates,f)

    with open("multibatching/datalocal/coordinates.json", "r") as f :
        coordinates = json.load(f)

    for current_transport_type in solution.problem.transport_types:
        name_transport_type = current_transport_type.name
        
        flows = []
        for flow in solution.list_flows:
            if flow.transport_link.transport_type == current_transport_type:
                l1 = flow.transport_link.location_l1
                l2 = flow.transport_link.location_l2

                # Store unique locations
                if l1.id not in locations_data:
                    locations_data[l1.id] = l1
                if l2.id not in locations_data:
                    locations_data[l2.id] = l2

                # Format hover text for the flow line
                products_str = "".join([f"- {p.name}: {qty}" for p, qty in flow.product_packing.items()])
                hover_text = (f"Flow: {l1.name} to {l2.name}"
                            f"Total Items: {flow.nb_packing}"
                            f"{products_str}")

                flows.append({
                    'start_lat': coordinates[l1.name][0], 'start_lon': coordinates[l1.name][1],
                    'end_lat': coordinates[l2.name][0], 'end_lon': coordinates[l2.name][1],
                    'quantity': flow.nb_packing,
                    'hover_text': hover_text
                })

        locations_df = pd.DataFrame([{
            'id': loc.id, 'name': loc.name, 'lat': coordinates[loc.name][0], 'lon': coordinates[loc.name][1],
            'supply': sum(loc.net_supply.values())
        } for loc in locations_data.values()])

        flows_df = pd.DataFrame(flows)

        # 2. Create the Plotly Figure
        fig = go.Figure()

        # 3. Add flow lines to the map
        for i, row in flows_df.iterrows():
            fig.add_trace(go.Scattergeo(
                mode="lines",
                lon=[row['start_lon'], row['end_lon']],
                lat=[row['start_lat'], row['end_lat']],
                line=dict(width=2 + row['quantity'] / 100, color='cornflowerblue'), # Width scales with quantity
                hoverinfo='text',
                hovertext=row['hover_text'],
                name=f"Flow {i+1}"
            ))

        # 4. Add location markers to the map
        fig.add_trace(go.Scattergeo(
            mode="markers",
            lon=locations_df['lon'],
            lat=locations_df['lat'],
            marker=dict(
                size=15,
                color=locations_df['supply'].apply(lambda x: 'green' if x >= 0 else 'red'),
                sizemin=4
            ),
            text=locations_df['name'],
            hoverinfo='text',
            hovertext=locations_df.apply(lambda r: f"{r['name']}Net Supply: {r['supply']}", axis=1)
        ))

        # 5. Configure the map layout
        fig.update_layout(
            title=name_transport_type,
            showlegend=True,
                width=900,  # Set the width in pixels
                height=600, # Set the height in pixels
            mapbox=dict(
                center=dict(lat=50, lon=10), # Center on Europe
            ),
            margin={"r":0,"t":40,"l":0,"b":0},
        )

        fig.show()

def interpolate_geodesic(start_lat, start_lon, end_lat, end_lon, num_points=20):
    """
    Generates intermediate points along the geodesic curve 
    between two coordinates.
    """
    geod = Geod(ellps='WGS84')
    
    points = geod.npts(
        lon1=start_lon, lat1=start_lat,
        lon2=end_lon, lat2=end_lat,
        npts=num_points
    )
    
    if not points or num_points < 2: 
        return [start_lat, end_lat], [start_lon, end_lon]
        
    lons_interp, lats_interp = zip(*points)
    
    # Convert tuples to lists to make them mutable
    lats = list(lats_interp)
    lons = list(lons_interp)
    
    # Manually overwrite the first and last points with the exact
    # start and end coordinates. This corrects any minor floating-point
    # discrepancies from the npts calculation and ensures a perfect connection.
    lats[0] = start_lat
    lons[0] = start_lon
    lats[-1] = end_lat
    lons[-1] = end_lon
    
    return lats, lons

def plot_solution_on_map_by_transport(solution: MultibatchingSolution):
    """
    Creates an interactive Plotly map of a MultibatchingSolution,
    grouping flows by route and transport type, with unique colors for each type.
    """
    # 1. Extract data, now including the transport type for each flow
    locations_data = {}
    with open("multibatching/datalocal/coordinates.json", "r") as f:
        coordinates = json.load(f)

    flows = []
    transport_types_in_solution = set() # To keep track of unique transport types

    for flow in solution.list_flows:
        
        l1 = flow.transport_link.location_l1
        l2 = flow.transport_link.location_l2

        transport_type_name = flow.transport_link.transport_type.name
        route = {}
        

        if "Ship" in transport_type_name:
            route_json = sr.searoute(
                [coordinates[l1.name][1], coordinates[l1.name][0]],
                [coordinates[l2.name][1], coordinates[l2.name][0]]
            )
            
            route_lats = [lat for lon, lat in route_json["geometry"]["coordinates"]]
            route_lons = [lon for lon, lat in route_json["geometry"]["coordinates"]]

            if route_lats:
                route_lats[0] = coordinates[l1.name][0]
                route_lons[0] = coordinates[l1.name][1]
                route_lats[-1] = coordinates[l2.name][0]
                route_lons[-1] = coordinates[l2.name][1]
            
            route["lat"] = route_lats
            route["lon"] = route_lons
        else:
            lats, lons = interpolate_geodesic(
                  start_lat=coordinates[l1.name][0],
                  start_lon=coordinates[l1.name][1],
                  end_lat=coordinates[l2.name][0],
                  end_lon=coordinates[l2.name][1],
                  num_points=50 
              )
            route["lat"] = lats
            route["lon"] = lons
        
        transport_types_in_solution.add(transport_type_name)

        if l1.id not in locations_data:
            locations_data[l1.id] = l1
        if l2.id not in locations_data:
            locations_data[l2.id] = l2

        products_str = "".join([f"\n  - {p.name}: {qty}" for p, qty in flow.product_packing.items()])
        flow_details = f"\n- Flow with {flow.nb_packing} items:{products_str}"

        flows.append({
            'start_name': l1.name, 'end_name': l2.name,
            'start_lat': coordinates[l1.name][0], 'start_lon': coordinates[l1.name][1],
            'end_lat': coordinates[l2.name][0], 'end_lon': coordinates[l2.name][1],
            'quantity': flow.nb_packing,
            'flow_details_text': flow_details,
            'transport_type': transport_type_name,
            'route':route # <-- ADDED THIS
        })

    locations_for_df = []
    for loc in locations_data.values():
        total_supply = sum(loc.net_supply.values())
        
        # 1. Filter to get outgoing (positive quantity) products
        outgoing_products = {p: q for p, q in loc.net_supply.items() if q > 0}
        
        # 2. Filter to get required (negative quantity) products
        required_products = {p: q for p, q in loc.net_supply.items() if q < 0}
        
        # 3. Build the base HTML hover text string
        hover_text = f"<b>{loc.name}</b><br>Net Supply: {total_supply}"
        
        # 4. Add the "Outgoing Products" section if it exists
        if outgoing_products:
            hover_text += "<br>--- Outgoing Products ---"
            for product, quantity in outgoing_products.items():
                hover_text += f"<br>- {product.name}: {quantity}"
                
        # 5. Add the "Required Products" section if it exists
        if required_products:
            hover_text += "<br>--- Required Products ---"
            for product, quantity in required_products.items():
                # Use abs() to display a readable, positive quantity
                hover_text += f"<br>- {product.name}: {abs(quantity)}"

        # 6. Append the complete dictionary to the list for the DataFrame
        locations_for_df.append({
            'id': loc.id,
            'name': loc.name,
            'lat': coordinates[loc.name][0],
            'lon': coordinates[loc.name][1],
            'supply': total_supply,
            'hovertext': hover_text
        })

    locations_df = pd.DataFrame(locations_for_df)

    if not flows:
        print("No flows to display.")
        return

    flows_df = pd.DataFrame(flows)

    # 2. Create the Plotly Figure and color scheme
    fig = go.Figure()
    
    # Create a dynamic color map for the transport types
    color_palette = qualitative.Plotly # A good default color sequence
    transport_color_map = {ttype: color_palette[i % len(color_palette)] 
                           for i, ttype in enumerate(sorted(list(transport_types_in_solution)))}

    # 3. Add grouped flow lines to the map
    # Group by route AND transport type
    grouped_flows = flows_df.groupby(['start_name', 'end_name', 'transport_type'])


    for (start_name, end_name, transport_type), group in grouped_flows:
        total_quantity = group['quantity'].sum()
        
        # Update hover text to include transport type
        combined_hover_text = (f"<b>{start_name} → {end_name}</b><br>"
                               f"Transport: {transport_type}<br>"
                               f"Total Items: {total_quantity}<br>"
                               "--- Details ---<br>" + 
                               "<br>".join(group['flow_details_text']))
        
        route_coords = group.iloc[0]
        print(route_coords)
        route_color = transport_color_map[transport_type] # Get the color for this type



        fig.add_trace(go.Scattergeo(
            mode="lines",
            lon=route_coords['route']['lon'],
            lat=route_coords['route']['lat'],
            line=dict(width=2 + total_quantity / 150, color=route_color), # <-- Use dynamic color
            hoverinfo='text',
            hovertext=combined_hover_text,
            name=f"{start_name} → {end_name} ({transport_type})", # <-- Updated legend name
            legendgroup=transport_type # This groups legend items by transport type
        ))

    # 4. Add location markers to the map
    fig.add_trace(go.Scattergeo(
        mode="markers",
        lon=locations_df['lon'],
        lat=locations_df['lat'],
        marker=dict(size=15, color=locations_df['supply'].apply(lambda x: 'green' if x >= 0 else 'red')),
        hoverinfo='text',
        hovertext=locations_df['hovertext'],
        name="Locations"
    ))

    # 5. Configure the map layout
    fig.update_layout(
        autosize=True,
        title_text="Solution Flows by Route and Transport Type",
        title_x=0.5,
        showlegend=True,
        width=2050, height=900,
        geo=dict(projection_type='mercator',
            center=dict(lat=48, lon=10),
            domain=dict(x=[0, 0.8], y=[0, 1])
        ),
        margin={"r":0,"t":40,"l":0,"b":0},

       legend=dict(
            x=0.65,         
            y=0.99,         
            xanchor='left', 
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='Black',
            borderwidth=1
        )
    )

    fig.show()

    
def plot_solution_on_map_grouped(solution: MultibatchingSolution):
    """
    Creates an interactive Plotly map of a MultibatchingSolution,
    grouping flows by route for a cleaner and more interactive legend.
    """
    # 1. Extract data from the solution object (similar to your original code)
    locations_data = {}
    with open("multibatching/datalocal/coordinates.json", "r") as f:
        coordinates = json.load(f)

    flows = []
    for flow in solution.list_flows:
        l1 = flow.transport_link.location_l1
        l2 = flow.transport_link.location_l2

        if l1.id not in locations_data:
            locations_data[l1.id] = l1
        if l2.id not in locations_data:
            locations_data[l2.id] = l2

        # Detailed info for this specific flow
        products_str = "".join([f"\n  - {p.name}: {qty}" for p, qty in flow.product_packing.items()])
        flow_details = f"\n- Flow with {flow.nb_packing} items:{products_str}"

        flows.append({
            'start_name': l1.name, 'end_name': l2.name,
            'start_lat': coordinates[l1.name][0], 'start_lon': coordinates[l1.name][1],
            'end_lat': coordinates[l2.name][0], 'end_lon': coordinates[l2.name][1],
            'quantity': flow.nb_packing,
            'flow_details_text': flow_details # Store individual flow details
        })

    locations_df = pd.DataFrame([{
        'id': loc.id, 'name': loc.name, 'lat': coordinates[loc.name][0], 'lon': coordinates[loc.name][1],
        'supply': sum(loc.net_supply.values())
    } for loc in locations_data.values()])

    if not flows:
        print("No flows to display.")
        return

    flows_df = pd.DataFrame(flows)

    # 2. Create the Plotly Figure
    fig = go.Figure()

    # 3. Add grouped flow lines to the map
    # Group by the route (start and end location names)
    grouped_flows = flows_df.groupby(['start_name', 'end_name'])

    for (start_name, end_name), group in grouped_flows:
        # Aggregate the information for this route
        total_quantity = group['quantity'].sum()
        combined_hover_text = (f"<b>Route: {start_name} to {end_name}</b><br>"
                               f"Total Items: {total_quantity}<br>"
                               "--- Details ---<br>" + 
                               "<br>".join(group['flow_details_text']))
        
        # Get coordinates from the first row of the group (they are all the same)
        route_coords = group.iloc[0]

        fig.add_trace(go.Scattergeo(
            mode="lines",
            lon=[route_coords['start_lon'], route_coords['end_lon']],
            lat=[route_coords['start_lat'], route_coords['end_lat']],
            line=dict(width=2 + total_quantity / 100, color='cornflowerblue'),
            hoverinfo='text',
            hovertext=combined_hover_text,
            name=f"{start_name} → {end_name}" # This creates a meaningful legend entry
        ))

    # 4. Add location markers to the map (same as your original code)
    fig.add_trace(go.Scattergeo(
        mode="markers",
        lon=locations_df['lon'],
        lat=locations_df['lat'],
        marker=dict(
            size=15,
            color=locations_df['supply'].apply(lambda x: 'green' if x >= 0 else 'red'),
            sizemin=4
        ),
        text=locations_df['name'],
        hoverinfo='text',
        hovertext=locations_df.apply(lambda r: f"<b>{r['name']}</b><br>Net Supply: {r['supply']}", axis=1),
        name="Locations" # Give the markers a name in the legend
    ))

    # 5. Configure the map layout (same as your original code)
    fig.update_layout(
        title_text="Solution Flows by Route",
        showlegend=True,
        width=900,
        height=600,
        geo=dict(
            scope='europe',
            projection_type='azimuthal equal area',
            center=dict(lat=48, lon=10),
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
    )

    fig.show()


def save_solution(solution : MultibatchingSolution, outputfile):
    list_sol = []
    for flow in solution.list_flows:
        dict_packing = {key.id : val for key,val in flow.product_packing.items()} 
        list_sol.append({"transport_link":flow.transport_link.id, "product_packing": dict_packing, "nb_packing": flow.nb_packing})

    with open(outputfile, 'w') as f :
        json.dump(list_sol, f)
    print("written")

def read_solution(problem: MultibatchingProblem, inputfile):
    with open(inputfile, "r") as f:
        json_data = json.load(f)
    packing_solution = []
    for elt in json_data:
        tl  = problem.transport_links_id_to_object[elt["transport_link"]]
        nb_packing = elt["nb_packing"]
        product_packing = {}
        for p,val in elt["product_packing"].items():
            if np.abs(int(val))>0:
                print(p,val)
                product_packing[problem.product_dict[p]] = val

        packing_solution.append(PackingTransport(tl, product_packing, nb_packing))
    return MultibatchingSolution(problem, list_flows=packing_solution)
        

    # packing_solution = []
    # for tl in json_data.keys():
        
    #         # type_link = tl.transport_type
    #         for b in range(len(self.batches[tl])):
    #             nb_packing = self.rp_model.getVarByName(f"x_{tl.id}_b{b}").X
    #             if nb_packing>0:
    #                 packing_solution.append(PackingTransport(transport_link=tl, product_packing=self.batches[tl][b], nb_packing=nb_packing))
    # return MultibatchingSolution(problem, list_flows)

def plotly_solution_graph(
        solution: MultibatchingSolution,
        problem: MultibatchingProblem,
        products_to_plot: Optional[List[Product]] = None,
        explode_flows: bool = True,
        use_logical_layout: bool = True,
        html_path: Optional[str] = None,
        png_path: Optional[str] = None,
        show_figure: bool = True
):
    """
    Renders and optionally saves product flows as an interactive Sankey diagram.

    Args:
        solution: The MultibatchingSolution to visualize.
        problem: The problem definition.
        products_to_plot: An optional list of products to display. If None, all products are shown.
        explode_flows: If True, uses intermediate nodes to separate overlapping flows for different products.
        use_logical_layout: If True, arranges nodes sequentially from left to right based on flow hierarchy.
        html_path (str, optional): The file path to save an interactive HTML file.
        png_path (str, optional): The file path to save a static PNG image. Requires 'kaleido' to be installed.
        show_figure (bool): If True, displays the figure in a browser/viewer.
    """
    # --- 1. Filter products and aggregate flows ---
    included_products = set(products_to_plot) if products_to_plot else set(problem.products)
    title_product_part = ", ".join(
        f"P{p.name}" for p in sorted(included_products, key=lambda x: x.id)) if products_to_plot else "All Products"

    aggregated_flows = defaultdict(float)
    for flow_item in solution.list_flows:
        for product, units in flow_item.product_packing.items():
            if product in included_products:
                total_units = units * flow_item.nb_packing
                if total_units > 1e-6:
                    from_loc_id = flow_item.transport_link.location_l1.id
                    to_loc_id = flow_item.transport_link.location_l2.id
                    aggregated_flows[(from_loc_id, to_loc_id, product, flow_item.transport_link.transport_type)] += (
                        total_units)

    if not aggregated_flows:
        print(f"No active flows found for the specified products: {title_product_part}")
        return

    # --- 2. Prepare node and link data structures ---
    product_colors = ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 'rgba(44, 160, 44, 0.8)',
                      'rgba(214, 39, 40, 0.8)', 'rgba(148, 103, 189, 0.8)', 'rgba(140, 86, 75, 0.8)']

    loc_id_to_loc = {loc.id: loc for loc in problem.locations}
    sorted_locations = sorted(problem.locations, key=lambda l: l.name)
    node_labels = [loc.name for loc in sorted_locations]
    node_colors = ['#333333'] * len(node_labels)
    node_to_index = {loc.id: i for i, loc in enumerate(sorted_locations)}
    sources, targets, values, link_colors, link_labels = [], [], [], [], []

    # --- 3. Generate Layout ---
    sankey_arrangement = 'snap'
    node_x, node_y = None, None

    if use_logical_layout:
        sankey_arrangement = 'freeform'
        node_x, node_y = [0.5] * len(node_labels), [0.95] * len(node_labels)  # Default for inactive nodes

        flow_graph = nx.DiGraph()
        for from_loc_id, to_loc_id, _ in aggregated_flows.keys():
            flow_graph.add_edge(from_loc_id, to_loc_id)

        node_levels = {}
        try:
            # CORRECTED: Properly calculate the level of each node in the flow sequence.
            sorted_nodes = list(nx.topological_sort(flow_graph))
            for node_id in sorted_nodes:
                max_pred_level = -1
                for pred_id in flow_graph.predecessors(node_id):
                    max_pred_level = max(max_pred_level, node_levels[pred_id])
                node_levels[node_id] = max_pred_level + 1

        except nx.NetworkXUnfeasible:
            print("⚠️ Warning: Flow graph contains cycles. Falling back to source-transshipment-sink layout.")
            loc_roles = {loc.id: 'transshipment' for loc in problem.locations}
            for loc in problem.locations:
                net_supply = sum(loc.net_supply.get(p, 0) for p in included_products)
                if net_supply > 1e-6:
                    loc_roles[loc.id] = 'source'
                elif net_supply < -1e-6:
                    loc_roles[loc.id] = 'sink'
            node_levels = {loc_id: 0 if role == 'source' else 1 if role == 'transshipment' else 2
                           for loc_id, role in loc_roles.items()}

        if node_levels:
            level_to_nodes = defaultdict(list)
            for node_id, level in node_levels.items():
                level_to_nodes[level].append(node_id)

            max_level = max(node_levels.values()) if node_levels else 0

            for level, ids_at_level in level_to_nodes.items():
                sorted_ids = sorted(ids_at_level, key=lambda id: loc_id_to_loc[id].name)
                for i, node_id in enumerate(sorted_ids):
                    if node_id in node_to_index:
                        idx = node_to_index[node_id]
                        node_x[idx] = 0.01 + 0.98 * (level / max_level) if max_level > 0 else 0.5
                        node_y[idx] = (i + 0.5) / len(sorted_ids)  # Spread vertically

    # --- 4. Generate Links ---
    for (from_loc_id, to_loc_id, product, transport_type), quantity in aggregated_flows.items():
        color = product_colors[problem.product_to_index[product] % len(product_colors)]
        from_loc_name = loc_id_to_loc[from_loc_id].name
        to_loc_name = loc_id_to_loc[to_loc_id].name
        hover_label = f"<b>{from_loc_name} → {to_loc_name}</b><br>Product: {product.name}<br>Quantity: {quantity:,.2f}"

        if explode_flows:
            intermediate_label = f"P{product.name}: {from_loc_name}→{to_loc_name}, {transport_type.name}"
            if intermediate_label not in node_to_index:
                node_labels.append(intermediate_label)
                node_colors.append(color.replace('0.8', '0.9'))
                node_to_index[intermediate_label] = len(node_labels) - 1
                if use_logical_layout:
                    from_idx = node_to_index[from_loc_id]
                    to_idx = node_to_index[to_loc_id]
                    node_x.append(np.mean([node_x[from_idx], node_x[to_idx]]))
                    node_y.append(np.mean([node_y[from_idx], node_y[to_idx]]))

            from_idx, intermediate_idx, to_idx = node_to_index[from_loc_id], node_to_index[intermediate_label], \
            node_to_index[to_loc_id]
            sources.extend([from_idx, intermediate_idx])
            targets.extend([intermediate_idx, to_idx])
            values.extend([quantity, quantity])
            link_colors.extend([color, color])
            link_labels.extend([hover_label, hover_label])
        else:
            sources.append(node_to_index[from_loc_id])
            targets.append(node_to_index[to_loc_id])
            values.append(quantity)
            link_colors.append(color)
            link_labels.append(hover_label)

    # --- 5. Create Figure ---
    fig = go.Figure(data=[go.Sankey(
        arrangement=sankey_arrangement,
        node=dict(
            pad=20, thickness=15, line=dict(color="black", width=0.5),
            label=node_labels, color=node_colors, x=node_x, y=node_y,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=sources, target=targets, value=values,
            label=link_labels, color=link_colors,
            hovertemplate='%{label}<extra></extra>'
        )
    )])

    fig.update_layout(
        title=dict(
            text=f"Logistics Flow Network: <b>{title_product_part}</b>",
            font=dict(
                size=50  # <-- Increase this value for a bigger title (e.g., 20)
            )
        ),

        # --- To change NODE and HOVER label size ---
        font=dict(
            family="Arial",
            size=50  # <-- Increase this value for bigger node/hover labels (e.g., 16)
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # --- 6. Save and/or Show ---
    if html_path:
        fig.write_html(html_path)
        print(f"✅ Interactive plot saved to: {os.path.abspath(html_path)}")
    if png_path:
        try:
            fig.write_image(png_path, width=1200, height=700, scale=2)
            print(f"✅ Static image saved to: {os.path.abspath(png_path)}")
        except ValueError as e:
            print(f"❌ Error saving PNG: {e}")
            print("👉 Please install kaleido: pip install kaleido")
    if show_figure:
        fig.show()


def generate_multibatching_problem(
        num_locations: int = 6,
        num_transport_types: int = 2,
        num_products: int = 2,
        min_dist: int = 1,
        max_dist: int = 10,
        min_capacity: int = 10,
        max_capacity: int = 20,
        min_product_size: int = 1,
        max_product_size: int = 5,
        min_product_value: int = 100,
        max_product_value: int = 1000,
        min_tt_cost: int = 10,
        max_tt_cost: int = 100,
        min_tt_speed: float = 0.1,
        max_tt_speed: float = 1.0,
        min_tt_emissions: int = 10,
        max_tt_emissions: int = 100,
        max_links_per_pair: int = 2,  # Max 2 links between any A->B pair (e.g., different transport types)
        demand_sparsity: float = 0.5,  # % of (location, product) pairs that will have non-zero demand
        max_demand_abs: int = 50,  # Max absolute supply/demand value for any product at a location
        seed: int = None  # For reproducibility
) -> MultibatchingProblem:
    """
    Generates a MultibatchingProblem instance with random data.

    Args:
        num_locations: Number of distinct locations.
        num_transport_types: Number of distinct transport resource types.
        num_products: Number of distinct products.
        min_dist, max_dist: Range for transport link distances.
        min_capacity, max_capacity: Range for transport type capacities.
        min_product_size, max_product_size: Range for product sizes.
        min_product_value, max_product_value: Range for product values.
        min_tt_cost, max_tt_cost: Range for transport type costs.
        min_tt_speed, max_tt_speed: Range for transport type speeds.
        min_tt_emissions: Range for transport type emissions.
        max_tt_emissions: Range for transport type emissions.
        max_links_per_pair: Maximum number of distinct transport types for links between any two locations.
        demand_sparsity: Proportion of (location, product) pairs that will have non-zero demand.
        max_demand_abs: Maximum absolute value for net supply/demand.
        seed: Seed for random number generator for reproducibility.

    Returns:
        An instance of MultibatchingProblem.
    """
    if seed is not None:
        random.seed(seed)

    # 1. Generate TransportType instances
    transport_types_list = []
    for i in range(num_transport_types):
        tt = TransportType(
            id=i + 1,
            cost=random.randint(min_tt_cost, max_tt_cost),
            speed=round(random.uniform(min_tt_speed, max_tt_speed), 1),
            emissions=random.randint(min_tt_emissions, max_tt_emissions),
            capacity=random.randint(min_capacity, max_capacity)
        )
        transport_types_list.append(tt)

    # 2. Generate Product instances
    products_list = []
    for i in range(num_products):
        # Each product can be transported by a random subset of transport types
        num_valid_tts = random.randint(1, len(transport_types_list))
        valid_transports = frozenset(random.sample(transport_types_list, num_valid_tts))
        prod = Product(
            id=i + 1,
            size=random.randint(min_product_size, max_product_size),
            value=random.randint(min_product_value, max_product_value),
            valid_transports=valid_transports
        )
        products_list.append(prod)

    # 3. Generate Location instances with balanced net_supply
    locations_list = []
    product_global_demands = defaultdict(int)  # To keep track of total supply/demand for each product

    # First, generate demands that sum to zero for each product
    # Create potential supply/demand points
    temp_demands = defaultdict(lambda: defaultdict(int))  # {loc_id: {prod: qty}}
    for i in range(num_locations):
        loc_id = f'l{i + 1}'
        for prod in products_list:
            if random.random() < demand_sparsity:  # Introduce sparsity
                demand_val = random.randint(-max_demand_abs, max_demand_abs)
                if demand_val != 0:
                    temp_demands[loc_id][prod] = demand_val
                    product_global_demands[prod] += demand_val

    # Adjust last location's demand to balance totals (simple balancing heuristic)
    if num_locations > 0:
        last_loc_id = f'l{num_locations}'
        for prod in products_list:
            if product_global_demands[prod] != 0:
                temp_demands[last_loc_id][prod] -= product_global_demands[prod]
                # Reset global demand after balancing
                product_global_demands[prod] = 0

    for i in range(num_locations):
        loc_id = f'l{i + 1}'
        loc = Location(id=loc_id, net_supply=temp_demands[loc_id], name=loc_id)
        locations_list.append(loc)

    # Check if total demand for each product is truly balanced
    for prod in products_list:
        total_net_supply = sum(loc.net_supply.get(prod, 0) for loc in locations_list)
        if total_net_supply != 0:
            # If not perfectly balanced (due to rounding/edge cases), this is where it would be
            # acknowledged. For this generator, we rely on the heuristic.
            pass

    # 4. Generate TransportLink instances
    transport_links_list = []
    current_link_id = 0
    for i in range(num_locations):
        for j in range(num_locations):
            if i == j:  # No links from a location to itself
                continue

            loc1 = locations_list[i]
            loc2 = locations_list[j]

            # Shuffle transport types to randomize which ones get chosen if max_links_per_pair < num_transport_types
            available_tts_for_pair = list(transport_types_list)
            random.shuffle(available_tts_for_pair)

            links_created_for_pair = 0
            for tt in available_tts_for_pair:
                if links_created_for_pair >= max_links_per_pair:
                    break  # Limit links per pair to max_links_per_pair distinct transport types

                # Check is implicitly handled by iterating through a shuffled list of unique transport types.
                # Each TT will be used at most once for a given (loc1, loc2) pair in this loop.

                link = TransportLink(
                    id=current_link_id,  # Assign a unique ID for each link
                    location_l1=loc1,
                    location_l2=loc2,
                    distance=round(random.uniform(min_dist, max_dist), 1),
                    transport_type=tt
                )
                transport_links_list.append(link)
                current_link_id += 1
                links_created_for_pair += 1

    # 5. Create MultibatchingProblem Instance
    problem_instance = MultibatchingProblem(
        transport_types=transport_types_list,
        products=products_list,
        locations=locations_list,
        transport_links=transport_links_list
    )

    return problem_instance




def basic_sanity_check(problem: MultibatchingProblem):
    products = problem.products
    total_flow_per_product = defaultdict(lambda: 0)
    details = defaultdict(lambda : [])
    for location in problem.locations:
        for p in location.net_supply:
            total_flow_per_product[p] += location.net_supply[p]
            details[p].append((location.id, location.net_supply[p]))

    for p in total_flow_per_product:
        print(p.id, total_flow_per_product[p])
        if total_flow_per_product[p] != 0:
            print("Problem", details[p])
    return total_flow_per_product



