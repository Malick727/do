from collections import defaultdict
from multibatching.problem import MultibatchingSolution, PackingTransport, TransportLink, MultibatchingProblem
from collections import defaultdict


def calculate_multibatch_percentage(solution: MultibatchingSolution) -> float:
    """
    Calculates the percentage of PackingTransport flows that are "multibatch".

    A flow is considered multibatch if it contains more than one product type
    in its packing.

    Args:
        solution (MultibatchingSolution): The solution to analyze.

    Returns:
        float: The percentage of multibatch flows (from 0.0 to 100.0).
    """
    total_flows = len(solution.list_flows)
    if total_flows == 0:
        return 0.0

    multibatch_count = 0
    for flow in solution.list_flows:
        # A packing is "multibatch" if it contains more than 1 product.
        if len(flow.product_packing) > 1:
            multibatch_count += 1

    return (multibatch_count / total_flows) * 100.0


def compare_solutions_visually(
        singlebatch_solution: MultibatchingSolution,
        multibatch_solution: MultibatchingSolution
):
    """
    Compares a single-batch and a multi-batch solution to highlight the
    efficiency gains of multibatching.

    It prints:
    1. A summary of overall metrics (costs, emissions, total trips).
    2. A detailed table showing which transport links were optimized,
       comparing the number of trips and packing configurations.

    Args:
        singlebatch_solution (MultibatchingSolution): The baseline solution using only single-product packing.
        multibatch_solution (MultibatchingSolution): The optimized solution using multi-product packing.
    """
    print("--- 🚀 Solution Efficiency Comparison ---")
    problem: MultibatchingProblem = singlebatch_solution.problem
    # --- 1. Overall Metrics Summary ---
    sb_eval = singlebatch_solution.problem.evaluate(singlebatch_solution)
    mb_eval = multibatch_solution.problem.evaluate(multibatch_solution)
    sb_trips = sum(flow.nb_packing for flow in singlebatch_solution.list_flows)
    mb_trips = sum(flow.nb_packing for flow in multibatch_solution.list_flows)

    print("\n✅ Overall Metrics:")
    print(f"| Metric          | Single-Batch        | Multi-Batch         | Improvement |")
    print(f"|-----------------|---------------------|---------------------|-------------|")
    print(f"| Total Trips 🚚   | {sb_trips:<20}| {mb_trips:<20}| {sb_trips - mb_trips:^11} |")
    for key in sb_eval:
        sb_val = sb_eval[key]
        mb_val = mb_eval[key]
        improvement = sb_val - mb_val
        print(f"| {key.title():<16}| {f'{sb_val:,.2f}':<20}| {f'{mb_val:,.2f}':<20}| {f'{improvement:,.2f}':^11} |")

    # --- 2. Detailed Link-by-Link Comparison ---
    sb_flows_by_link = defaultdict(list)
    for flow in singlebatch_solution.list_flows:
        sb_flows_by_link[flow.transport_link].append(flow)

    mb_flows_by_link = defaultdict(list)
    for flow in multibatch_solution.list_flows:
        mb_flows_by_link[flow.transport_link].append(flow)

    all_links = set(sb_flows_by_link.keys()) | set(mb_flows_by_link.keys())

    print("\n✅ Detailed Trip Consolidation on Transport Links:")
    print(
        f"| Transport Link                      | Single-Batch Details              | Multi-Batch Details               | Trip Reduction |")
    print(
        f"|-------------------------------------|-----------------------------------|-----------------------------------|----------------|")

    for link in sorted(all_links, key=lambda l: (l.location_l1.name, l.location_l2.name)):
        sb_link_flows = sb_flows_by_link.get(link, [])
        mb_link_flows = mb_flows_by_link.get(link, [])

        sb_total_trips = sum(f.nb_packing for f in sb_link_flows)
        mb_total_trips = sum(f.nb_packing for f in mb_link_flows)

        # Only show links where multibatching made a change
        if sb_total_trips == mb_total_trips and len(sb_link_flows) == len(mb_link_flows):
            continue

        link_name = f"{link.location_l1.name}->{link.location_l2.name} ({link.transport_type.name})"

        sb_details_parts = [
            f"({f.nb_packing}x {{ {', '.join([f'P{p.name}:{u}' for p, u in f.product_packing.items()])} }})" for f in
            sb_link_flows]
        sb_details = f"{sb_total_trips} trips: {' '.join(sb_details_parts)}"

        mb_details_parts = [
            f"({f.nb_packing}x {{ {', '.join([f'P{p.name}:{u}' for p, u in f.product_packing.items()])} }})" for f in
            mb_link_flows]
        mb_details = f"{mb_total_trips} trips: {' '.join(mb_details_parts)}"

        reduction = sb_total_trips - mb_total_trips

        print(f"| {link_name:<35} | {sb_details:<33} | {mb_details:<33} | {reduction:^14} |")

    print("\n--- End of Comparison ---\n")




def compare_solutions_by_route(
        singlebatch_solution: MultibatchingSolution,
        multibatch_solution: MultibatchingSolution
):
    """
    Compares two solutions at a route level (location-to-location),
    highlighting efficiency gains through trip reduction and transport utilization.

    It prints:
    1. A summary of overall metrics including average fill rate.
    2. A detailed table showing trips and fill rates for each route.

    Args:
        singlebatch_solution (MultibatchingSolution): The baseline solution.
        multibatch_solution (MultibatchingSolution): The optimized solution.
    """
    print("--- 🚚 Route-Level Efficiency Comparison ---")

    def get_route_metrics(solution: MultibatchingSolution):
        """Helper to calculate trips, volume, and capacity for each route."""
        metrics_by_route = defaultdict(lambda: {"trips": 0, "volume": 0, "capacity": 0})
        for flow in solution.list_flows:
            route = (flow.transport_link.location_l1.name, flow.transport_link.location_l2.name)

            packed_volume = sum(
                p.size * u for p, u in flow.product_packing.items()
            ) * flow.nb_packing

            transport_capacity = flow.transport_link.transport_type.capacity * flow.nb_packing

            metrics_by_route[route]["trips"] += flow.nb_packing
            metrics_by_route[route]["volume"] += packed_volume
            metrics_by_route[route]["capacity"] += transport_capacity

        return metrics_by_route

    sb_route_metrics = get_route_metrics(singlebatch_solution)
    mb_route_metrics = get_route_metrics(multibatch_solution)

    # --- 1. Overall Metrics Summary ---
    sb_eval = singlebatch_solution.problem.evaluate(singlebatch_solution)
    mb_eval = multibatch_solution.problem.evaluate(multibatch_solution)
    sb_total_trips = sum(m["trips"] for m in sb_route_metrics.values())
    mb_total_trips = sum(m["trips"] for m in mb_route_metrics.values())

    sb_total_volume = sum(m["volume"] for m in sb_route_metrics.values())
    sb_total_capacity = sum(m["capacity"] for m in sb_route_metrics.values())
    sb_avg_fill = (sb_total_volume / sb_total_capacity * 100) if sb_total_capacity > 0 else 0

    mb_total_volume = sum(m["volume"] for m in mb_route_metrics.values())
    mb_total_capacity = sum(m["capacity"] for m in mb_route_metrics.values())
    mb_avg_fill = (mb_total_volume / mb_total_capacity * 100) if mb_total_capacity > 0 else 0

    print("\n✅ Overall Metrics:")
    print(f"| Metric                | Single-Batch         | Multi-Batch          | Improvement  |")
    print(f"|-----------------------|----------------------|----------------------|--------------|")
    print(
        f"| Total Trips 🚚         | {sb_total_trips:<21}| {mb_total_trips:<21}| {sb_total_trips - mb_total_trips:^12} |")
    print(
        f"| Avg. Fill Rate 📦     | {f'{sb_avg_fill:.1f}%':<21}| {f'{mb_avg_fill:.1f}%':<21}| {f'{mb_avg_fill - sb_avg_fill:+.1f}%':^12} |")
    for key in sb_eval:
        sb_val, mb_val = sb_eval[key], mb_eval[key]
        print(f"| {key.title():<22}| {f'{sb_val:,.2f}':<21}| {f'{mb_val:,.2f}':<21}| {f'{sb_val - mb_val:,.2f}':^12} |")

    # --- 2. Detailed Route-by-Route Comparison ---
    all_routes = sorted(list(set(sb_route_metrics.keys()) | set(mb_route_metrics.keys())))

    print("\n✅ Detailed Route Performance:")
    print(
        f"| Route                         | SB Trips | MB Trips | SB Avg. Fill Rate | MB Avg. Fill Rate | Trip Reduction |")
    print(
        f"|-------------------------------|----------|----------|-------------------|-------------------|----------------|")

    for route in all_routes:
        sb_metrics = sb_route_metrics.get(route, {"trips": 0, "volume": 0, "capacity": 0})
        mb_metrics = mb_route_metrics.get(route, {"trips": 0, "volume": 0, "capacity": 0})

        # Only show routes that have changed
        if sb_metrics["trips"] == mb_metrics["trips"]:
            continue

        route_name = f"{route[0]} -> {route[1]}"

        sb_fill = (sb_metrics["volume"] / sb_metrics["capacity"] * 100) if sb_metrics["capacity"] > 0 else 0
        mb_fill = (mb_metrics["volume"] / mb_metrics["capacity"] * 100) if mb_metrics["capacity"] > 0 else 0

        reduction = sb_metrics["trips"] - mb_metrics["trips"]

        print(
            f"| {route_name:<29} | {sb_metrics['trips']:^8} | {mb_metrics['trips']:^8} | {f'{sb_fill:.1f}%':^17} | {f'{mb_fill:.1f}%':^17} | {reduction:^14} |")

    print("\n--- End of Comparison ---\n")