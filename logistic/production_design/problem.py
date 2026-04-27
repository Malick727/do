"""
Quantified Global Logistics Problem - Flow-based formulation

Extension of the GlobalLogisticProblem where:
- Parts have demand quantities (not just 0/1 decisions)
- Production and transport decisions are continuous (quantities)
- Costs are per-unit (not fixed activation costs)
- BOM requirements are quantity-based (N parents need N of each child)
- Flow conservation at each location for each part

This paves the way toward multi-batching and realistic production planning.
"""

import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np

from logistic.network_design.problem import CountryConstraint, Location, Part, TransportMean, TransportOption, \
    GlobalLogisticSolution, GlobalLogisticProblem

from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
    ObjectiveRegister,
    EncodingRegister,
    ModeOptim,
    ObjectiveHandling,
    ObjectiveDoc,
    TypeObjective,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# 1. Extended Domain Models for Quantified Problem
# =====================================================================

class QuantifiedPart(Part):
    """
    Part with quantity demand.

    Extends Part with:
    - demand: Total quantity needed (for final products) or minimum to produce (INTEGER)
    - unit_value_added: Value added per unit (used for country/location constraints)
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.demand: int = 0  # Total demand for this part (INTEGER)
        self.unit_value_added: float = 0.0  # Value added per unit (replaces value_added)

    def __repr__(self):
        return f"QuantifiedPart({self.name}, demand={self.demand})"

    def __reduce__(self):
        """Support for pickling."""
        # Return a callable and its arguments that can reconstruct this object
        return (
            self.__class__,
            (self.name,),
            self.__dict__  # This will restore all attributes
        )

    def __setstate__(self, state):
        """Restore state during unpickling."""
        self.__dict__.update(state)


class QuantifiedLocation(Location):
    """
    Location with per-unit production costs.

    Extends Location with:
    - unit_production_cost: Dict mapping Part -> cost per unit
    - max_quantity_per_part: Optional capacity constraints per part (INTEGER)
    """
    def __init__(self, name: str):
        super().__init__(name)
        # Cost to produce one unit of each part at this location
        self.unit_production_cost: Dict[Part, float] = {}
        # Optional: maximum quantity that can be produced for each part (INTEGER)
        self.max_quantity_per_part: Dict[Part, int] = {}

    def __repr__(self):
        roles = []
        if self.is_production:
            roles.append("Prod")
        if self.is_warehouse:
            roles.append("WH")
        role_str = "/".join(roles) if roles else "Unknown"
        return f"QuantifiedLocation({self.name} - {role_str}, {len(self.unit_production_cost)} parts)"

    def __reduce__(self):
        """Support for pickling."""
        return (
            self.__class__,
            (self.name,),
            self.__dict__
        )

    def __setstate__(self, state):
        """Restore state during unpickling."""
        self.__dict__.update(state)


class QuantifiedTransportOption(TransportOption):
    """
    Transport option with per-unit costs.

    Extends TransportOption where:
    - unit_transport_cost: Cost per unit shipped (total_cost becomes unit cost)
    """
    # The dataclass is frozen, so we can't add new fields directly
    # Instead, we interpret total_cost as cost per unit
    pass


# =====================================================================
# 2. Quantified Solution
# =====================================================================

class QuantifiedGlobalLogisticSolution(Solution):
    """
    Solution for quantified global logistics problem.

    Instead of binary decisions, stores:
    - production_quantities: How much of each part is produced at each location (INTEGER)
    - flow_quantities: How much of each part flows through each transport option (INTEGER)
    """

    def __init__(
        self,
        problem: "QuantifiedGlobalLogisticProblem",
        production_quantities: Dict[Tuple[Part, Location], int],
        flow_quantities: Dict[TransportOption, int],
        **kwargs: Any
    ):
        """
        Initialize solution.

        Args:
            problem: The problem instance
            production_quantities: Dict mapping (Part, Location) -> quantity produced (INTEGER)
            flow_quantities: Dict mapping TransportOption -> quantity shipped (INTEGER)
        """
        super().__init__(problem)
        self.problem = problem
        self.production_quantities = production_quantities
        self.flow_quantities = flow_quantities
        self.kwargs = kwargs

    def __str__(self) -> str:
        lines = ["\n" + "=" * 80]
        lines.append(" QUANTIFIED GLOBAL LOGISTICS SOLUTION")
        lines.append("=" * 80)

        lines.append("\n📦 PRODUCTION QUANTITIES:")
        lines.append("-" * 60)

        # Group by part
        production_by_part = defaultdict(list)
        for (part, loc), qty in sorted(self.production_quantities.items(), key=lambda x: x[1], reverse=True):
            if qty > 0:
                production_by_part[part].append((loc, qty))

        for part in sorted(production_by_part.keys(), key=lambda p: p.name):
            lines.append(f"\n  {part.name}:")
            for loc, qty in production_by_part[part]:
                lines.append(f"    • {loc.name}: {qty:,.1f} units")

        lines.append("\n🚢 TRANSPORT FLOWS:")
        lines.append("-" * 60)

        if not self.flow_quantities or all(q == 0 for q in self.flow_quantities.values()):
            lines.append("  (No transport flows)")
        else:
            total_flow_cost = 0.0
            for transport_option, qty in sorted(self.flow_quantities.items(),
                                               key=lambda x: x[1], reverse=True):
                if qty > 0:
                    # total_cost is per-unit cost in quantified version
                    flow_cost = transport_option.total_cost * qty
                    total_flow_cost += flow_cost
                    lines.append(
                        f"  • {transport_option.part.name}: "
                        f"{transport_option.origin.name} → {transport_option.destination.name}\n"
                        f"      Quantity: {qty:,.1f}, via {transport_option.mean.name}, "
                        f"Cost: €{flow_cost:,.2f}"
                    )
            lines.append("-" * 60)
            lines.append(f"  Total Transport Cost: €{total_flow_cost:,.2f}")

        lines.append("=" * 80 + "\n")
        return "\n".join(lines)

    def copy(self) -> "QuantifiedGlobalLogisticSolution":
        return QuantifiedGlobalLogisticSolution(
            problem=self.problem,
            production_quantities=self.production_quantities.copy(),
            flow_quantities=self.flow_quantities.copy()
        )

    def lazy_copy(self) -> "QuantifiedGlobalLogisticSolution":
        return self.copy()


# =====================================================================
# 3. Quantified Problem Definition
# =====================================================================

class QuantifiedGlobalLogisticProblem(Problem):
    """
    Global logistics problem with quantity-based decisions.

    Key differences from base problem:
    - Parts have demand quantities
    - Production and transport are continuous variables
    - Costs are per-unit, not fixed
    - Flow conservation constraints at each location
    - BOM requirements are quantity-based
    """

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister({})

    def get_solution_type(self) -> type[Solution]:
        return QuantifiedGlobalLogisticSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "production_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
                "transport_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
                "nrc_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
                "total_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
            }
        )

    def __init__(self, data_folder: str):
        """
        Initialize from the same data format as GlobalLogisticProblem,
        but interpret it as a quantified problem.
        """
        # First, load using the base class parser
        base_problem = GlobalLogisticProblem(data_folder)

        # Convert to quantified versions
        self.data_folder = data_folder

        # Convert Parts to QuantifiedParts
        self.parts: List[QuantifiedPart] = []
        self.parts_dict: Dict[str, QuantifiedPart] = {}

        for part in base_problem.parts:
            q_part = QuantifiedPart(part.name)
            q_part.value_added = part.value_added
            q_part.unit_value_added = part.value_added  # Assume value_added is per-unit
            q_part.parent_part = None  # Will be set later
            q_part.child_parts = []
            q_part.sourcing_requirement = part.sourcing_requirement
            q_part.produceable_at = set()  # Will be populated
            q_part.demand = 0.0  # Will be set for final products

            self.parts.append(q_part)
            self.parts_dict[part.name] = q_part

        # Rebuild BOM structure
        for part in base_problem.parts:
            q_part = self.parts_dict[part.name]
            if part.parent_part:
                q_part.parent_part = self.parts_dict[part.parent_part.name]
            q_part.child_parts = [self.parts_dict[c.name] for c in part.child_parts]

        # Set demand for final products (parts with no parent)
        for part in self.parts:
            if part.parent_part is None:
                part.demand = 1  # Default demand (INTEGER), can be overridden

        # Convert Locations to QuantifiedLocations
        self.locations: Dict[str, QuantifiedLocation] = {}

        for loc in base_problem.locations:
            q_loc = QuantifiedLocation(loc.name)
            q_loc.country = loc.country
            q_loc.factory_activation_cost = loc.factory_activation_cost
            q_loc.is_production = loc.is_production
            q_loc.is_warehouse = loc.is_warehouse
            q_loc.min_value_added = loc.min_value_added
            q_loc.max_value_added = loc.max_value_added

            # Default unit production cost (can be refined based on location cost)
            # Use factory_activation_cost / 100 as a heuristic for per-unit cost
            base_unit_cost = max(1.0, loc.factory_activation_cost / 100.0)
            q_loc.unit_production_cost = {}

            self.locations[loc.name] = q_loc

        # Update produceable_at relationships
        for part in self.parts:
            base_part = next(p for p in base_problem.parts if p.name == part.name)
            part.produceable_at = {self.locations[loc.name] for loc in base_part.produceable_at}

            # Set unit production cost for each produceable location
            # Formula: unit_cost = (factory_activation_cost / 10) * (1 + value_added_factor)
            # This ensures more expensive factories have higher per-unit costs
            # Dividing by 10 (not 100) keeps unit costs as meaningful integers
            for loc in part.produceable_at:
                if loc.is_production:
                    # Base cost proportional to factory activation cost (keep as integer-friendly)
                    base_cost = max(10.0, loc.factory_activation_cost / 10.0)
                    # Adjust by part value (more valuable parts cost more to produce)
                    value_factor = 1.0 + part.unit_value_added / 1000.0
                    loc.unit_production_cost[part] = base_cost * value_factor

        # Convert TransportOptions
        self.transport_options: List[QuantifiedTransportOption] = []

        for t_opt in base_problem.transport_options:
            q_origin = self.locations[t_opt.origin.name]
            q_dest = self.locations[t_opt.destination.name]
            q_part = self.parts_dict[t_opt.part.name]

            # Create QuantifiedTransportOption
            # In the quantified version, total_cost is interpreted as cost per unit
            q_t_opt = QuantifiedTransportOption(
                origin=q_origin,
                destination=q_dest,
                part=q_part,
                mean=t_opt.mean,
                duration=t_opt.duration,
                total_cost=t_opt.total_cost  # This is now cost per unit
            )
            self.transport_options.append(q_t_opt)

        # Transport means (same as base)
        self.transport_means: List[TransportMean] = base_problem.transport_means

        # Country constraints (same as base)
        self.country_constraints: List[CountryConstraint] = base_problem.country_constraints

        # Build location to country mapping
        self.location_to_country = {loc.name: loc.country for loc in self.locations.values()}

    def evaluate(self, solution: QuantifiedGlobalLogisticSolution) -> Dict[str, float]:
        """
        Evaluate solution based on:
        - Factory activation cost: one-time cost for each location used
        - Production cost: sum of (unit_cost * quantity) for all production
        - Transport cost: sum of (unit_cost * quantity) for all flows
        - NRC cost: one-time cost for each transport mean used

        Note: Quantities are integers, but costs are floats.
        """
        factory_activation_cost = 0.0
        production_cost = 0.0
        transport_cost = 0.0
        nrc_cost = 0.0

        # Factory activation cost (one-time for each location with production)
        active_locations = set()
        for (part, loc), qty in solution.production_quantities.items():
            if qty > 0:
                active_locations.add(loc)

        for loc in active_locations:
            factory_activation_cost += loc.factory_activation_cost

        # Production cost (per-unit)
        for (part, loc), qty in solution.production_quantities.items():
            if qty > 0:
                unit_cost = loc.unit_production_cost.get(part, 0.0)
                production_cost += unit_cost * float(qty)

        # Transport cost
        for transport_option, qty in solution.flow_quantities.items():
            if qty > 0:
                # total_cost is per-unit cost in quantified version
                transport_cost += transport_option.total_cost * float(qty)

        # NRC cost (one-time for each mean used)
        used_means = set()
        for transport_option, qty in solution.flow_quantities.items():
            if qty > 0:
                used_means.add(transport_option.mean)

        for mean in used_means:
            nrc_cost += mean.nrc_cost

        total_cost = factory_activation_cost + production_cost + transport_cost + nrc_cost

        return {
            "factory_activation_cost": factory_activation_cost,
            "production_cost": production_cost,
            "transport_cost": transport_cost,
            "nrc_cost": nrc_cost,
            "total_cost": total_cost,
        }

    def satisfy(self, solution: QuantifiedGlobalLogisticSolution, verbose: bool = False) -> bool:
        """
        Check if solution satisfies all constraints:
        1. Demand satisfaction for final products
        2. Flow conservation at each location
        3. BOM requirements (if producing N parents, need N of each child)
        4. Non-negativity
        5. Country/location value-added constraints
        6. Sourcing requirements (minimum number of active locations per part)

        Note: All quantities are integers, so exact equality checks (no epsilon needed for quantities).
        """
        violations = []

        # --- Constraint 1: Demand satisfaction ---
        for part in self.parts:
            if part.demand > 0:
                total_production = sum(
                    qty for (p, loc), qty in solution.production_quantities.items()
                    if p == part
                )
                if total_production != part.demand:  # Exact integer comparison
                    violations.append(
                        f"Demand violation for {part.name}: "
                        f"produced {total_production}, needed {part.demand}"
                    )

        # --- Constraint 2: Flow conservation ---
        # For each (part, location): incoming + produced = consumed + outgoing
        # Special case: For final products (with demand), they satisfy external demand

        for part in self.parts:
            for loc in self.locations.values():
                # Incoming flow
                incoming = sum(
                    qty for t_opt, qty in solution.flow_quantities.items()
                    if t_opt.part == part and t_opt.destination == loc
                )

                # Produced at this location
                produced = solution.production_quantities.get((part, loc), 0)

                # Outgoing flow
                outgoing = sum(
                    qty for t_opt, qty in solution.flow_quantities.items()
                    if t_opt.part == part and t_opt.origin == loc
                )

                # Consumed (for BOM assembly)
                consumed = 0
                # If this location produces any parent parts, they consume child parts
                for parent in self.parts:
                    if part in parent.child_parts:
                        parent_qty = solution.production_quantities.get((parent, loc), 0)
                        consumed += parent_qty  # Need 1 child per parent

                # For final products (with demand), production satisfies external demand
                # So we don't check flow conservation - they can be "consumed" by demand
                if part.demand > 0 and part.parent_part is None:
                    # Final product - skip flow conservation, already checked in demand constraint
                    continue

                # Balance check (exact integer comparison)
                available = incoming + produced
                used = outgoing + consumed

                if available != used:
                    violations.append(
                        f"Flow conservation violation for {part.name} at {loc.name}: "
                        f"available={available}, used={used}, diff={available - used}"
                    )

        # --- Constraint 3: BOM requirements (implicit in flow conservation) ---
        # Already checked above

        # --- Constraint 4: Non-negativity ---
        for (part, loc), qty in solution.production_quantities.items():
            if qty < 0:
                violations.append(f"Negative production: {part.name} at {loc.name} = {qty}")

        for transport_option, qty in solution.flow_quantities.items():
            if qty < 0:
                violations.append(
                    f"Negative flow: {transport_option.part.name} "
                    f"{transport_option.origin.name} -> {transport_option.destination.name} = {qty}"
                )

        # --- Constraint 5: Country value-added constraints ---
        # Value added is float (unit_value_added * integer quantity)
        epsilon = 1e-6  # Only for value-added comparisons (not quantities)
        value_by_country = defaultdict(float)
        for (part, loc), qty in solution.production_quantities.items():
            if qty > 0:
                country = loc.country
                value_by_country[country] += part.unit_value_added * float(qty)

        for constraint in self.country_constraints:
            country_value = value_by_country.get(constraint.country, 0.0)
            if constraint.min_value_added is not None:
                if country_value < constraint.min_value_added - epsilon:
                    violations.append(
                        f"Country constraint violation: {constraint.country} "
                        f"has {country_value:.2f}, needs min {constraint.min_value_added:.2f}"
                    )
            if constraint.max_value_added is not None:
                if country_value > constraint.max_value_added + epsilon:
                    violations.append(
                        f"Country constraint violation: {constraint.country} "
                        f"has {country_value:.2f}, max allowed {constraint.max_value_added:.2f}"
                    )

        # --- Constraint 6: Location value-added constraints ---
        value_by_location = defaultdict(float)
        for (part, loc), qty in solution.production_quantities.items():
            if qty > 0:
                value_by_location[loc] += part.unit_value_added * float(qty)

        for loc in self.locations.values():
            loc_value = value_by_location.get(loc, 0.0)
            if loc.min_value_added is not None:
                if loc_value < loc.min_value_added - epsilon:
                    violations.append(
                        f"Location constraint violation: {loc.name} "
                        f"has {loc_value:.2f}, needs min {loc.min_value_added:.2f}"
                    )
            if loc.max_value_added is not None:
                if loc_value > loc.max_value_added + epsilon:
                    violations.append(
                        f"Location constraint violation: {loc.name} "
                        f"has {loc_value:.2f}, max allowed {loc.max_value_added:.2f}"
                    )

        # --- Constraint 7: Sourcing requirements ---
        for part in self.parts:
            if part.sourcing_requirement > 1:
                active_locations = set(
                    loc for (p, loc), qty in solution.production_quantities.items()
                    if p == part and qty > 0  # Integer check
                )
                if len(active_locations) < part.sourcing_requirement:
                    violations.append(
                        f"Sourcing requirement violation: {part.name} "
                        f"produced at {len(active_locations)} locations, "
                        f"needs {part.sourcing_requirement}"
                    )

        # Report violations
        if violations and verbose:
            logger.warning(f"Solution has {len(violations)} constraint violations:")
            for v in violations[:10]:  # Show first 10
                logger.warning(f"  - {v}")
            if len(violations) > 10:
                logger.warning(f"  ... and {len(violations) - 10} more")

        return len(violations) == 0

    def get_dummy_solution(self) -> QuantifiedGlobalLogisticSolution:
        """
        Generate a simple feasible solution:
        - Produce all final products at first available location
        - Recursively produce all children at same location (no transport)
        """
        production_quantities = {}
        flow_quantities = {}

        def allocate_production(part: QuantifiedPart, location: QuantifiedLocation, quantity: int):
            """Recursively allocate production for a part and its children at a location."""
            # Add production for this part
            current_qty = production_quantities.get((part, location), 0)
            production_quantities[(part, location)] = current_qty + quantity

            # Recursively produce children at same location
            for child in part.child_parts:
                if location in child.produceable_at:
                    allocate_production(child, location, quantity)
                else:
                    # Child not produceable at this location - this shouldn't happen in dummy solution
                    # but we'll handle it gracefully by finding another location
                    if child.produceable_at:
                        child_loc = min(child.produceable_at, key=lambda loc: loc.name)
                        allocate_production(child, child_loc, quantity)
                        # TODO: Add transport flow from child_loc to location

        # Find final products (parts with demand > 0)
        for part in self.parts:
            if part.demand > 0:
                if part.produceable_at:
                    # Produce all at first available location
                    first_loc = min(part.produceable_at, key=lambda loc: loc.name)
                    allocate_production(part, first_loc, part.demand)

        return QuantifiedGlobalLogisticSolution(
            problem=self,
            production_quantities=production_quantities,
            flow_quantities=flow_quantities
        )


if __name__ == "__main__":
    import os

    # Example usage
    this_folder = os.path.dirname(__file__)
    data_folder = os.path.join(this_folder, "../global_logistic/data")

    print("Loading quantified problem...")
    problem = QuantifiedGlobalLogisticProblem(data_folder)

    print(f"\nParts: {len(problem.parts)}")
    for part in problem.parts[:5]:
        print(f"  - {part.name}: demand={part.demand}, value_added={part.unit_value_added}")

    print(f"\nLocations: {len(problem.locations)}")
    for loc_name, loc in list(problem.locations.items())[:3]:
        n_parts = len(loc.unit_production_cost)
        print(f"  - {loc_name}: {n_parts} parts produceable")

    print(f"\nTransport options: {len(problem.transport_options)}")

    # Generate dummy solution
    print("\nGenerating dummy solution...")
    solution = problem.get_dummy_solution()

    print(f"\nProduction locations:")
    for (part, loc), qty in sorted(solution.production_quantities.items(), key=lambda x: x[1], reverse=True)[:10]:
        if qty > 0:
            print(f"  {part.name} at {loc.name}: {qty:.1f} units")

    # Evaluate
    objectives = problem.evaluate(solution)
    print(f"\nObjectives:")
    for key, value in objectives.items():
        print(f"  {key}: {value:,.2f}")

    # Validate
    is_feasible = problem.satisfy(solution, verbose=True)
    print(f"\nSolution feasible: {is_feasible}")
