"""Verbatim extract of SA5.py lines 1625-1691 (DeliveryRouteOptimizer) and of the
VRP block of SA5.evaluate_solution (lines 876-917). NOT used by parcelnet; kept
only for transparency. Known defects are listed in experimental/README.md.
Requires ortools (not installed in the validation environment; never executed)."""
from typing import List, Tuple

import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # Google OR-Tools for VRP

class DeliveryRouteOptimizer:
    def __init__(self, distance_matrix: np.ndarray, drop_off_time: float, max_shift_time: float):
        self.distance_matrix = distance_matrix
        self.drop_off_time = drop_off_time  # Time per parcel drop-off (in hours)
        self.max_shift_time = max_shift_time  # Maximum shift time per driver (in hours)

    def solve_vrp(self, service_point_idx: int, customer_indices: List[int], parcel_counts: List[int]) -> Tuple[int, float, float]:
        """Solve the VRP for a given service point."""
        num_customers = len(customer_indices)
        num_locations = num_customers + 1  # Include service point as depot

        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)  # 1 vehicle, depot at index 0

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node, to_node] * 1000)  # Convert to meters

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add time constraint
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = self.distance_matrix[from_node, to_node] / 50  # Assume 50 km/h average speed
            return int((travel_time + (self.drop_off_time if to_node != 0 else 0)) * 60)  # Convert to minutes

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            0,  # No slack
            int(self.max_shift_time * 60),  # Max shift time in minutes
            True,  # Start cumul to zero
            "Time"
        )

        # Solve the problem
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            return float('inf'), float('inf'), float('inf')  # No feasible solution

        # Extract results
        total_distance = 0
        total_time = 0
        num_drivers = 1

        for vehicle_id in range(1):  # Single vehicle for now
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                next_index = solution.Value(routing.NextVar(index))
                total_distance += self.distance_matrix[manager.IndexToNode(index), manager.IndexToNode(next_index)]
                total_time += time_callback(index, next_index)
                index = next_index

        return num_drivers, total_distance / 1000, total_time / 60  # Convert to km and hours


# ---- Block from SA5.evaluate_solution (lines 876-917), shown for reference only ----
VRP_BLOCK_IN_EVALUATE = r"""
        # VRP-related costs
        total_drivers = 0
        total_kilometers = 0
        total_drop_off_time = 0

        for sp_idx in np.where(solution['y'])[0]:  # Iterate over open service points
            sp_id = self.candidate_locations[sp_idx]
            sp_node_idx = self.node_to_idx[sp_id]

            # Get customers assigned to this service point
            assigned_customers = [
                cust_idx for cust_idx in range(len(self.customer_nodes))
                if solution['x_assigned'][cust_idx, sp_idx]
            ]

            if not assigned_customers:
                continue

            customer_indices = [self.node_to_idx[self.customer_nodes[cust_idx]] for cust_idx in assigned_customers]
            parcel_counts = [self.demand_data[self.customer_nodes[cust_idx]]['delivery'] for cust_idx in assigned_customers]

            # Solve VRP for this service point
            vrp_solver = DeliveryRouteOptimizer(
                self.distance_matrix,
                self.parameters['DROP_OFF_TIME_MINUTES'] / 60,  # Convert to hours
                8  # Max shift time in hours
            )
            drivers, kilometers, drop_off_time = vrp_solver.solve_vrp(sp_node_idx, customer_indices, parcel_counts)

            total_drivers += drivers
            total_kilometers += kilometers
            total_drop_off_time += drop_off_time

        # Calculate delivery costs
        total_driving_cost = total_kilometers * self.parameters['VEHICLE_COST_PER_KM']
        total_drop_off_cost = total_drop_off_time * self.parameters['COST_PER_PARCEL_DROP_OFF']
        total_driver_labor_cost = (
            min(total_drivers, 10) * self.parameters['COST_DRIVER_PERMANENT_DAY'] +
            max(0, total_drivers - 10) * self.parameters['COST_DRIVER_SUBCONTRACTED_DAY']
        )

        delivery_costs = total_driving_cost + total_drop_off_cost + total_driver_labor_cost
"""
