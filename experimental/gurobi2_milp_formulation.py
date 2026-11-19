import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import math
import networkx as nx
from typing import Dict, List, Tuple
import time

class OptimizedServicePointOptimizer:
    """
    Optimized version for better performance on limited hardware.
    Key improvements:
    1. Reduced search space through preprocessing
    2. Efficient constraint formulation
    3. Better memory management
    4. Heuristic initialization
    """
    
    def __init__(self):
        self.model = None
        self.decision_vars = {}
        self.results = {}
        self.preprocessed_data = {}
        
    def preprocess_data(self, 
                        customer_demands: Dict[int, float],
                        distances: Dict[Tuple[int, int], float],
                        potential_locations: List[int]):  # max_locations_per_customer dropped
        """
        Preprocess data without filtering by maximum distance.
        For each customer, all potential locations with available distances are considered.
        """
        print("Preprocessing data without filtering by max distance...")
        
        # Convert distances to km if in meters
        distances_km = {k: v/1000 if v > 100 else v for k, v in distances.items()}
        
        # For each customer, consider all locations that have an associated distance
        customer_location_pairs = {}
        for customer in customer_demands.keys():
            candidate_locations = []
            candidate_distances = []
            
            for location in potential_locations:
                if (customer, location) in distances_km:
                    candidate_locations.append(location)
                    candidate_distances.append(distances_km[(customer, location)])
            
            if candidate_locations:
                # Sort candidates by distance (ascending)
                sorted_pairs = sorted(zip(candidate_locations, candidate_distances), key=lambda x: x[1])
                # Use all sorted candidate locations (dropping max_locations_per_customer limitation)
                candidate_list = [loc for loc, _ in sorted_pairs]
                customer_location_pairs[customer] = candidate_list
            else:
                # Fallback to the top 5 closest among all potential locations
                all_dists = [(loc, distances_km.get((customer, loc), float('inf'))) 
                             for loc in potential_locations]
                sorted_dists = sorted(all_dists, key=lambda x: x[1])
                customer_location_pairs[customer] = [loc for loc, _ in sorted_dists[:5]]
        
        # Identify the aggregate set of candidate locations used by any customer
        used_locations = set()
        for locs in customer_location_pairs.values():
            used_locations.update(locs)
        used_locations = list(used_locations)
        
        print(f"Reduced from {len(potential_locations)} to {len(used_locations)} potential locations")
        print(f"Average locations per customer: {np.mean([len(locs) for locs in customer_location_pairs.values()]):.1f}")
        
        self.preprocessed_data = {
            'customer_demands': customer_demands,
            'distances_km': distances_km,
            'customer_location_pairs': customer_location_pairs,
            'used_locations': used_locations,
            'original_locations': potential_locations
        }
            
    def setup_model(self, 
                   f_sp: float = 50000,
                   c_cap: float = 36.50,
                   c_hd: float = 1.50,
                   br_sp: float = 0.02,
                   br_cw: float = 0.01,
                   big_m: float = 200000):
        """
        Set up the optimization model with preprocessed data.
        """
        if not self.preprocessed_data:
            raise ValueError("Must call preprocess_data() first")
            
        # Store parameters
        self.I = list(self.preprocessed_data['customer_demands'].keys())
        self.J = self.preprocessed_data['used_locations']
        self.D = self.preprocessed_data['customer_demands']
        self.dist_km = self.preprocessed_data['distances_km']
        self.customer_locations = self.preprocessed_data['customer_location_pairs']
        self.F_SP = f_sp
        self.C_cap = c_cap
        self.C_hd = c_hd
        self.BR_sp = br_sp
        self.BR_cw = br_cw
        self.M = big_m
        
        print(f"Creating model with {len(self.I)} customers and {len(self.J)} potential locations")
        
        # Create the model with performance settings
        self.model = gp.Model("Optimized_Service_Point")
        self.model.Params.OutputFlag = 1
        self.model.Params.Threads = 4  # Limit threads for older hardware
        self.model.Params.Method = 2   # Use barrier method for LP relaxations
        self.model.Params.Presolve = 2  # Aggressive presolve
        
        # Decision Variables (only create needed ones)
        self._create_decision_variables()
        
        # Objective Function
        self._set_objective()
        
        # Constraints
        self._add_constraints()
    
    def _create_decision_variables(self):
        """Create decision variables. Assignment variables are now continuous [0,1]."""
        
        # SP opening decisions remain binary
        self.decision_vars['y'] = {
            j: self.model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
            for j in self.J
        }
        
        # SP capacities
        self.decision_vars['c'] = {
            j: self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"c_{j}")
            for j in self.J
        }
        
        # Customer assignments: continuous fraction variables in [0,1]
        self.decision_vars['x_p'] = {}
        self.decision_vars['x_h'] = {}
        
        for customer in self.I:
            for location in self.customer_locations.get(customer, []):
                self.decision_vars['x_p'][(customer, location)] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_p_{customer}_{location}")
                self.decision_vars['x_h'][(customer, location)] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_h_{customer}_{location}")
        
        # Bounced parcels
        self.decision_vars['b'] = {
            j: self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"b_{j}")
            for j in self.J
        }
        
        self.model.update()
    
    def _set_objective(self):
        """Set the objective function."""
        fixed_costs = gp.quicksum(self.F_SP * self.decision_vars['y'][j] for j in self.J)
        capacity_costs = gp.quicksum(self.C_cap * self.decision_vars['c'][j] for j in self.J)
        
        # Modified delivery cost: cost per delivered parcel (1.5) times the customer demand fraction delivered.
        delivery_costs = gp.quicksum(
            self.C_hd * self.D[customer] *
            self.decision_vars['x_h'][(customer, location)]
            for customer in self.I 
            for location in self.customer_locations.get(customer, [])
            if (customer, location) in self.decision_vars['x_h']
        )
        
        self.model.setObjective(fixed_costs + capacity_costs + delivery_costs, GRB.MINIMIZE)
    
    def _add_constraints(self):
        """Add constraints with revised assignment formulation."""
        
        # For each customer, force a split: 65% pickup and 35% delivery
        for customer in self.I:
            valid_locations = self.customer_locations.get(customer, [])
            if valid_locations:
                pickup_vars = [self.decision_vars['x_p'][(customer, loc)]
                               for loc in valid_locations
                               if (customer, loc) in self.decision_vars['x_p']]
                delivery_vars = [self.decision_vars['x_h'][(customer, loc)]
                                 for loc in valid_locations
                                 if (customer, loc) in self.decision_vars['x_h']]
                
                self.model.addConstr(gp.quicksum(pickup_vars) == 0.65,
                                     name=f"pickup_fraction_{customer}")
                self.model.addConstr(gp.quicksum(delivery_vars) == 0.35,
                                     name=f"delivery_fraction_{customer}")
        
        # SP activation constraints
        for (customer, location), var in self.decision_vars['x_p'].items():
            self.model.addConstr(var <= self.decision_vars['y'][location],
                                 name=f"sp_act_p_{customer}_{location}")
        
        for (customer, location), var in self.decision_vars['x_h'].items():
            self.model.addConstr(var <= self.decision_vars['y'][location],
                                 name=f"sp_act_h_{customer}_{location}")
        
        # Capacity constraints & bounce rate constraints remain unchanged...
        for location in self.J:
            self.model.addConstr(
                self.decision_vars['c'][location] <= self.M * self.decision_vars['y'][location],
                name=f"capacity_{location}"
            )
            pickup_demand = gp.quicksum(
                self.D[customer] * self.decision_vars['x_p'][(customer, location)]
                for customer in self.I if (customer, location) in self.decision_vars['x_p']
            )
            self.model.addConstr(
                self.decision_vars['c'][location] >= (1 - self.BR_sp) * pickup_demand,
                name=f"bounce_rate_{location}"
            )
            self.model.addConstr(
                self.decision_vars['b'][location] >= pickup_demand - self.decision_vars['c'][location],
                name=f"bounced_def_{location}"
            )
        
        # City-wide bounce rate
        total_bounced = gp.quicksum(self.decision_vars['b'][j] for j in self.J)
        total_pickup_demand = gp.quicksum(self.D[customer] * var 
                                          for var in self.decision_vars['x_p'].values())
        self.model.addConstr(total_bounced <= self.BR_cw * total_pickup_demand,
                             name="city_bounce_rate")
        
    def add_heuristic_start(self):
        """Add a heuristic starting solution to warm-start the solver."""
        print("Adding heuristic starting solution...")
        
        # Simple greedy heuristic: open locations with highest demand density
        location_demands = {}
        for location in self.J:
            total_demand = sum(
                self.D[customer] 
                for customer in self.I 
                if location in self.customer_locations.get(customer, [])
            )
            location_demands[location] = total_demand
        
        # Sort locations by demand and open top ones
        sorted_locations = sorted(location_demands.items(), key=lambda x: x[1], reverse=True)
        num_to_open = min(len(self.J) // 4, 20)  # Open 25% or max 20
        
        for i, (location, _) in enumerate(sorted_locations[:num_to_open]):
            self.decision_vars['y'][location].Start = 1
            self.decision_vars['c'][location].Start = 4000  # Some reasonable capacity
    
    def solve(self, time_limit=3600):
        """Solve with performance optimizations."""
        print(f"Starting optimization with {time_limit}s time limit...")
        
        # Set solver parameters for older hardware
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', 0.08)  # Accept 5% gap
        self.model.setParam('MIPFocus', 3)   # Focus on finding good solutions quickly

        
        start_time = time.time()
        self.model.optimize()
        solve_time = time.time() - start_time
        
        print(f"Solve time: {solve_time:.1f} seconds")
        
        if self.model.Status == GRB.OPTIMAL:
            print("Optimal solution found!")
            self._extract_results()
            return True
        elif self.model.Status == GRB.TIME_LIMIT:
            if self.model.SolCount > 0:
                print(f"Time limit reached, but found solution with gap: {self.model.MIPGap*100:.2f}%")
                self._extract_results()
                return True
            else:
                print("Time limit reached, no solution found")
                return False
        else:
            print(f"Solver status: {self.model.Status}")
            return False
    
    def _extract_results(self):
        """Extract results from solved model."""
        self.results = {
            'objective_value': self.model.ObjVal,
            'mip_gap': getattr(self.model, 'MIPGap', 0) * 100,
            'opened_sps': [],
            'sp_capacities': {},
            'customer_assignments': {},
            'total_bounced': 0
        }
        
        # Extract opened SPs
        for location in self.J:
            if self.decision_vars['y'][location].X > 0.5:
                self.results['opened_sps'].append(location)
                self.results['sp_capacities'][location] = self.decision_vars['c'][location].X
        
        # Extract customer assignments
        for customer in self.I:
            for location in self.customer_locations.get(customer, []):
                if (customer, location) in self.decision_vars['x_p'] and \
                   self.decision_vars['x_p'][(customer, location)].X > 0.5:
                    self.results['customer_assignments'][customer] = ('pickup', location)
                    break
                elif (customer, location) in self.decision_vars['x_h'] and \
                     self.decision_vars['x_h'][(customer, location)].X > 0.5:
                    self.results['customer_assignments'][customer] = ('delivery', location)
                    break
        
        # Extract bounced parcels
        self.results['total_bounced'] = sum(
            self.decision_vars['b'][j].X for j in self.J
        )
    
    def get_summary(self):
        """Return optimization summary."""
        if not self.results:
            return "No solution available."
        
        pickup_count = sum(1 for assignment in self.results['customer_assignments'].values() 
                          if assignment[0] == 'pickup')
        delivery_count = len(self.results['customer_assignments']) - pickup_count
        
        return f"""

Optimization Results Summary:
============================

Total Annual Cost: €{self.results['objective_value']:,.2f}
MIP Gap: {self.results['mip_gap']:.2f}%
Service Points Opened: {len(self.results['opened_sps'])}
Locations: {self.results['opened_sps'][:10]}{'...' if len(self.results['opened_sps']) > 10 else ''}
Total Capacity: {sum(self.results['sp_capacities'].values()):.0f} parcels/day
Total Bounced: {self.results['total_bounced']:.2f}

Assignment Summary:
  - Pickup: {pickup_count} customers
  - Delivery: {delivery_count} customers
        """

def prepare_optimized_data():
    """Load and prepare data for optimization."""
    print("Loading data...")
    xls = pd.ExcelFile('data_Maastricht_2024.xlsx')
    nodes_df = pd.read_excel(xls, 'Nodes')
    
    # Adjusted: set each customer's demand correctly for a total annual demand of 1.4M
    # Daily total demand = 1,400,000 / 365, and then equally distributed among nodes/customers.
    daily_total_demand = 1400000 / 365
    num_customers = len(nodes_df)
    avg_daily_demand_per_customer = daily_total_demand / num_customers  # ≈ 0.48
    
    customer_demands = {row['NODE ID']: avg_daily_demand_per_customer for _, row in nodes_df.iterrows()}
    potential_locations = nodes_df['NODE ID'].tolist()
    
    # Load precomputed distances
    print("Loading distances...")
    df = pd.read_csv('shortest_paths_optimized_1000m.csv')
    distances = {(int(row['source']), int(row['target'])): float(row['distance_m'])
                 for _, row in df.iterrows()}
    
    return customer_demands, distances, potential_locations

def main():
    """Main optimization function."""
    customer_demands, distances, potential_locations = prepare_optimized_data()
    
    optimizer = OptimizedServicePointOptimizer()
    
    # Preprocess to reduce problem size
    optimizer.preprocess_data(
        customer_demands=customer_demands,
        distances=distances,
        potential_locations=potential_locations
    )
    
    # Setup and solve
    optimizer.setup_model(
        f_sp=50000,
        c_cap=36.50,
        c_hd=1.50,
        br_sp=0.02,
        br_cw=0.01
    )
    
    if optimizer.solve(time_limit= 1800):  # 30 minutes
        print(optimizer.get_summary())
    else:
        print("Could not find a solution within time limit")

if __name__ == "__main__":
    main()