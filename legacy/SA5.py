import pandas as pd
import numpy as np
import time
import random
import math
import pickle
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedServicePointOptimizer:
    def __init__(self):
        # Core parameters
        self.parameters = {
            'F_SP': 50000,     # Fixed cost per service point per year (€)
            'C_cap': 36.50,    # Yearly capacity cost per parcel (€0.10/day * 365)
            'C_hd': 1.50,      # Home delivery cost per km per parcel (€)
            'BR_sp': 0.02,     # Max bounce rate per service point (2%)
            'BR_cw': 0.01,     # Max city-wide bounce rate (1%)
            'M': 200000        # Max capacity per service point
        }
        
        # APL-specific parameters
        self.apl_fixed_cost = 500  # Fixed cost per APL per year (€)
        self.apl_capacity = 60      # Capacity per APL (parcels/day)
        
        # Demand split
        self.pickup_ratio = 0.57
        self.delivery_ratio = 0.43
        
        # Data structures
        self.nodes_data = None
        self.demand_data = None
        self.distance_matrix = None
        self.demographic_data = None
        
        # Solution tracking
        self.current_solution = None
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        
        # Optimization settings
        self.candidate_locations = []
        self.customer_nodes = []
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.apl_locations = []  # List of APL locations

        # Driver cost parameters
        self.parameters.update({
            'COST_DRIVER_PERMANENT_DAY': 150,  # Cost per day for permanent driver (€)
            'COST_DRIVER_SUBCONTRACTED_DAY': 250,  # Cost per day for subcontracted driver (€)
            'VEHICLE_COST_PER_KM': 0.21,  # Vehicle cost per km (€)
            'DRIVER_HOURLY_WAGE': 150 / 8,  # Hourly wage for permanent driver (€18.75/hour)
            'DROP_OFF_TIME_MINUTES': 3,  # Time per parcel drop-off (minutes)
        })

        # Calculate per-parcel drop-off cost
        self.parameters['COST_PER_PARCEL_DROP_OFF'] = (
            self.parameters['DRIVER_HOURLY_WAGE'] / 60 * self.parameters['DROP_OFF_TIME_MINUTES']
        )

    def load_data(self, excel_file: str, distance_file: str, assignments_file: Optional[str] = None):
        """Load all required data from Excel and CSV files."""
        print("Loading data from files...")
        
        # Load Excel data
        excel_data = pd.ExcelFile(excel_file)
        
        # Load nodes
        self.nodes_data = excel_data.parse('Nodes')
        print(f"Loaded {len(self.nodes_data)} nodes")
        
        # Load demographic data for demand estimation
        try:
            self.demographic_data = excel_data.parse('CBS Squares')
            print(f"Loaded demographic data for {len(self.demographic_data)} squares")
        except:
            print("Warning: Could not load CBS demographic data")
        
        # Load service point data if available
        try:
            service_points_data = excel_data.parse('Service Point Locations')
            # Skip header row and get only the Location ID column values
            self.candidate_locations = service_points_data['Location ID'].dropna().astype(int).tolist()
            print(f"Using {len(self.candidate_locations)} predefined service point locations")
        except Exception as e:
            print(f"Could not load service point locations: {e}")
            # Use all nodes as potential locations
            self.candidate_locations = self.nodes_data['NODE ID'].tolist()
            print(f"Using all {len(self.candidate_locations)} nodes as potential locations")
    
        # All nodes are potential customers
        self.customer_nodes = self.nodes_data['NODE ID'].tolist()
        
        # Load APL locations or assign random ones
        try:
            apl_data = excel_data.parse('APL Locations')
            self.apl_locations = apl_data['Location ID'].dropna().astype(int).tolist()
            print(f"Loaded {len(self.apl_locations)} predefined APL locations")
        except Exception as e:
            print(f"Could not load APL locations: {e}")
            # Assign 20 random APLs from candidate locations
            available_for_apls = [loc for loc in self.candidate_locations]
            if len(available_for_apls) >= 20:
                self.apl_locations = random.sample(available_for_apls, 20)
                print(f"Assigned 20 random APL locations from candidate nodes: {self.apl_locations}")
            else:
                self.apl_locations = available_for_apls[:min(20, len(available_for_apls))]
                print(f"Assigned {len(self.apl_locations)} APL locations (limited by available candidates)")
        
        # Create mapping dictionaries - INCLUDE APL LOCATIONS
        all_nodes = sorted(set(self.customer_nodes + self.candidate_locations + self.apl_locations))
        self.node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Load distances with caching
        self._load_distances_cached(distance_file)
        
        # Estimate demand based on demographics or use uniform distribution
        self._estimate_demand()
        
        # Load existing assignments if provided
        if assignments_file:
            self._load_existing_assignments(assignments_file)
        
        # Add validation
        self._validate_data()
        
        excel_data.close()

    def _load_distances_cached(self, distance_file: str):
        """Load distance matrix with intelligent caching."""
        # Generate cache filename based on nodes and distance file
        needed_nodes = sorted(set(self.node_to_idx.keys()))
        cache_key = f"{len(needed_nodes)}_{min(needed_nodes)}_{max(needed_nodes)}"
        cache_filename = f"distance_cache_{cache_key}.pkl"
        
        # Check if cache exists and is newer than source file
        cache_exists = os.path.exists(cache_filename)
        cache_valid = False
        
        if cache_exists:
            try:
                cache_time = os.path.getmtime(cache_filename)
                source_time = os.path.getmtime(distance_file)
                cache_valid = cache_time > source_time
            except:
                cache_valid = False
        
        if cache_valid:
            print(f"Loading cached distance matrix from {cache_filename}...")
            try:
                with open(cache_filename, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Validate cache matches current nodes
                if (cache_data['node_to_idx'] == self.node_to_idx and 
                    cache_data['distance_matrix'].shape == (len(needed_nodes), len(needed_nodes))):
                    
                    self.distance_matrix = cache_data['distance_matrix']
                    print(f"Successfully loaded cached distance matrix ({self.distance_matrix.shape})")
                    return
                else:
                    print("Cache validation failed - nodes don't match. Reloading...")
            except Exception as e:
                print(f"Error loading cache: {e}. Reloading...")
        
        # Load fresh distance matrix
        print("Loading distance matrix with optimization...")
        start_time = time.time()
        
        # Get only the nodes we actually need
        needed_nodes_set = set(self.node_to_idx.keys())
        
        # Try edge-list format first (much faster)
        sample_df = pd.read_csv(distance_file, nrows=5)
        
        if all(col in sample_df.columns for col in ['source', 'target', 'distance_m']):
            self._load_edge_list_optimized(distance_file, needed_nodes_set)
        else:
            self._load_matrix_optimized(distance_file, needed_nodes_set)
        
        load_time = time.time() - start_time
        print(f"Distance loading completed in {load_time:.1f}s")
        
        # Save to cache
        try:
            print(f"Saving distance matrix to cache ({cache_filename})...")
            cache_data = {
                'distance_matrix': self.distance_matrix,
                'node_to_idx': self.node_to_idx,
                'created_time': time.time(),
                'source_file': distance_file
            }
            
            with open(cache_filename, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Distance matrix cached successfully")
            
        except Exception as e:
            print(f"Warning: Could not save distance cache: {e}")

    def _load_distances(self, distance_file: str):
        """Optimized distance loading with pre-filtering and sparse storage."""
        print("Loading distance matrix with optimization...")
        start_time = time.time()
        
        # Get only the nodes we actually need
        needed_nodes = set(self.node_to_idx.keys())
        
        # Try edge-list format first (much faster)
        sample_df = pd.read_csv(distance_file, nrows=5)
        
        if all(col in sample_df.columns for col in ['source', 'target', 'distance_m']):
            self._load_edge_list_optimized(distance_file, needed_nodes)
        else:
            self._load_matrix_optimized(distance_file, needed_nodes)
        
        print(f"Distance loading completed in {time.time() - start_time:.1f}s")

    def _load_edge_list_optimized(self, distance_file: str, needed_nodes: set):
        """Ultra-fast edge-list loading with filtering."""
        # Pre-filter distances > 6km (most customers won't use distant service points)
        MAX_DISTANCE_KM = 6.0
        
        # Use chunked reading for memory efficiency
        chunk_size = 100000
        n_nodes = len(self.node_to_idx)
        self.distance_matrix = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
        
        processed = 0
        for chunk in pd.read_csv(distance_file, chunksize=chunk_size):
            # Filter chunk to only needed nodes and reasonable distances
            chunk = chunk[
                (chunk['source'].isin(needed_nodes)) &
                (chunk['target'].isin(needed_nodes)) &
                (chunk['distance_m'] <= MAX_DISTANCE_KM * 1000)
            ]
            
            # Vectorized assignment
            for _, row in chunk.iterrows():
                src_idx = self.node_to_idx[int(row['source'])]
                tgt_idx = self.node_to_idx[int(row['target'])]
                distance_km = float(row['distance_m']) / 1000
                
                self.distance_matrix[src_idx, tgt_idx] = distance_km
                self.distance_matrix[tgt_idx, src_idx] = distance_km  # Symmetric
            
            processed += len(chunk)
            if processed % 50000 == 0:
                print(f"Processed {processed:,} distance pairs...")
    
        np.fill_diagonal(self.distance_matrix, 0.0)

    def _load_matrix_optimized(self, distance_file: str, needed_nodes: set):
        """Optimized matrix loading with column pre-filtering and path completion."""
        # Read header to identify relevant columns
        header_df = pd.read_csv(distance_file, nrows=0)

        # Convert all column names to strings for consistent comparison
        all_columns = [str(col) for col in header_df.columns]
        needed_nodes_str = {str(n) for n in needed_nodes}

        # Find relevant columns (excluding the first index column)
        relevant_cols = [col for col in all_columns[1:] if col in needed_nodes_str]
        relevant_cols = [all_columns[0]] + relevant_cols  # Include index column

        print(f"Loading {len(relevant_cols)-1}/{len(all_columns)-1} relevant columns")

        try:
            # Load only needed columns and rows
            matrix_df = pd.read_csv(
                distance_file, 
                index_col=0,
                usecols=relevant_cols,
                dtype={col: np.float32 for col in relevant_cols[1:]}  # Specify float32 for distance columns
            )
            
            # Convert index to string for consistent comparison
            matrix_df.index = matrix_df.index.astype(str)
            matrix_df.columns = matrix_df.columns.astype(str)
            
            # Filter to relevant rows
            relevant_rows = [str(idx) for idx in matrix_df.index if str(idx) in needed_nodes_str]
            matrix_df = matrix_df.loc[relevant_rows]
            
            # Vectorized matrix filling
            n_nodes = len(self.node_to_idx)
            self.distance_matrix = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
            
            # Create index mappings for faster lookup with string conversion
            row_mapping = {}
            col_mapping = {}
            
            for idx in matrix_df.index:
                try:
                    node_id = int(idx)
                    if node_id in self.node_to_idx:
                        row_mapping[str(idx)] = self.node_to_idx[node_id]
                except (ValueError, TypeError):
                    continue
            
            for col in matrix_df.columns:
                try:
                    node_id = int(col)
                    if node_id in self.node_to_idx:
                        col_mapping[str(col)] = self.node_to_idx[node_id]
                except (ValueError, TypeError):
                    continue
            
            # Fill distance matrix with error handling
            for row_str, row_idx in row_mapping.items():
                for col_str, col_idx in col_mapping.items():
                    try:
                        val = matrix_df.loc[row_str, col_str]
                        if pd.notna(val) and val != 'inf' and val != float('inf'):
                            distance_km = float(val) / 1000  # Convert to km
                            if distance_km >= 0:  # Ensure non-negative
                                self.distance_matrix[row_idx, col_idx] = distance_km
                    except (KeyError, ValueError, TypeError) as e:
                        # Skip problematic entries
                        continue
            
            np.fill_diagonal(self.distance_matrix, 0.0)
            
            # CRITICAL FIX: Complete missing paths using shortest path approximation
            print("Completing missing distance paths...")
            self._complete_missing_paths()
            
        except Exception as e:
            print(f"Error loading matrix format: {e}")
            print("Falling back to uniform distances...")
            # Fallback: create uniform distance matrix
            n_nodes = len(self.node_to_idx)
            self.distance_matrix = np.full((n_nodes, n_nodes), 2.0, dtype=np.float32)  # 2km default
            np.fill_diagonal(self.distance_matrix, 0.0)

    def _complete_missing_paths(self):
        """Complete missing paths in distance matrix using various fallback strategies."""
        n_nodes = self.distance_matrix.shape[0]
        
        # Count infinite values
        inf_count = np.sum(np.isinf(self.distance_matrix))
        total_elements = n_nodes * n_nodes - n_nodes  # Exclude diagonal
        
        print(f"Found {inf_count} infinite distances out of {total_elements} total pairs")
        
        if inf_count == 0:
            return
        
        # Strategy 1: Make matrix symmetric (if one direction exists, use it for both)
        print("Step 1: Making matrix symmetric...")
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.isinf(self.distance_matrix[i, j]) and not np.isinf(self.distance_matrix[j, i]):
                    self.distance_matrix[i, j] = self.distance_matrix[j, i]
                elif not np.isinf(self.distance_matrix[i, j]) and np.isinf(self.distance_matrix[j, i]):
                    self.distance_matrix[j, i] = self.distance_matrix[i, j]
        
        # Count remaining infinite values
        inf_count_after_symmetry = np.sum(np.isinf(self.distance_matrix))
        print(f"After symmetry: {inf_count_after_symmetry} infinite distances remain")
        
        if inf_count_after_symmetry == 0:
            return
        
        # Strategy 2: Fill missing values with 2km (default assumption)
        print("Step 2: Filling missing values with default distance (2km)...")
        default_distance = 2.0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if np.isinf(self.distance_matrix[i, j]):
                    self.distance_matrix[i, j] = default_distance
                    self.distance_matrix[j, i] = default_distance  # Symmetric
        
        # Final validation
        final_inf_count = np.sum(np.isinf(self.distance_matrix))
        print(f"Final result: {final_inf_count} infinite distances remaining")
        
        if final_inf_count > 0:
            print("WARNING: Still have infinite distances - this may cause optimization issues")

    def _estimate_demand(self):
        """Estimate daily parcel demand per node based on demographics with pickup/delivery split."""
        print("Estimating demand per node...")
        
        # Total annual demand: 1.4M parcels
        annual_demand = 1400000
        daily_demand = annual_demand / 365
        
        self.demand_data = {}
        
        if self.demographic_data is not None:
            print("Using demographic-based demand estimation...")
            # Use demographic data to weight demand
            demand_weights = {}
            
            try:
                # Merge nodes with demographic data
                nodes_with_demo = self.nodes_data.merge(
                    self.demographic_data, 
                    left_on='SQUARE', 
                    right_on='Square', 
                    how='left'
                )
                
                # Weight by population and household characteristics
                for _, row in nodes_with_demo.iterrows():
                    node_id = row['NODE ID']
                    
                    # Handle missing demographic data with defaults
                    population = row.get('Population', 100) if pd.notna(row.get('Population', 100)) else 100
                    households = row.get('Households', population * 0.4) if pd.notna(row.get('Households', population * 0.4)) else population * 0.4
                    
                    # Ensure positive values
                    population = max(population, 1)
                    households = max(households, 1)
                    
                    # Higher demand for areas with more households and higher income
                    base_weight = households
                    income_multiplier = 1.0
                    
                    high_income_pct = row.get('Percentage high income households', 0)
                    if pd.notna(high_income_pct) and high_income_pct > 0:
                        income_multiplier = 1.0 + (high_income_pct / 100 * 0.5)  # Up to 50% more demand
                    
                    final_weight = base_weight * income_multiplier
                    demand_weights[node_id] = max(final_weight, 1.0)  # Ensure minimum weight
                
                # Ensure all customer nodes have weights
                for node_id in self.customer_nodes:
                    if node_id not in demand_weights or pd.isna(demand_weights[node_id]):
                        demand_weights[node_id] = 1.0  # Default weight
                
                # Remove any NaN weights
                demand_weights = {k: v for k, v in demand_weights.items() if not pd.isna(v) and v > 0}
                
                # Normalize weights
                total_weight = sum(demand_weights.values())
                if total_weight > 0:
                    print(f"Successfully calculated demographic weights for {len(demand_weights)} nodes")
                    for node_id in self.customer_nodes:
                        weight = demand_weights.get(node_id, 1.0)
                        total_node_demand = (weight / total_weight) * daily_demand
                        
                        # Ensure positive values
                        total_node_demand = max(total_node_demand, 0.01)
                        
                        self.demand_data[node_id] = {
                            'total': total_node_demand,
                            'pickup': total_node_demand * self.pickup_ratio,
                            'delivery': total_node_demand * self.delivery_ratio
                        }
                else:
                    print("Warning: No valid demographic weights found, using uniform distribution")
                    self._create_uniform_demand(daily_demand)
                    
            except Exception as e:
                print(f"Error in demographic demand estimation: {e}")
                print("Falling back to uniform distribution")
                self._create_uniform_demand(daily_demand)
        else:
            print("No demographic data available, using uniform distribution")
            self._create_uniform_demand(daily_demand)
        
        # Final validation and cleanup
        self._validate_and_fix_demand_data(daily_demand)

    def _create_uniform_demand(self, daily_demand):
        """Create uniform demand distribution across all customer nodes."""
        demand_per_node = daily_demand / len(self.customer_nodes)
        
        for node_id in self.customer_nodes:
            self.demand_data[node_id] = {
                'total': demand_per_node,
                'pickup': demand_per_node * self.pickup_ratio,
                'delivery': demand_per_node * self.delivery_ratio
            }

    def _validate_and_fix_demand_data(self, expected_daily_demand):
        """Validate demand data and fix any issues."""
        print("Validating and cleaning demand data...")
        
        # Check for NaN values and fix them
        nodes_to_fix = []
        for node_id, demand_dict in self.demand_data.items():
            for key, value in demand_dict.items():
                if pd.isna(value) or np.isnan(value) or np.isinf(value) or value < 0:
                    nodes_to_fix.append(node_id)
                    break
        
        if nodes_to_fix:
            print(f"Found {len(nodes_to_fix)} nodes with invalid demand values - fixing...")
            
            # Calculate average from valid nodes
            valid_demands = []
            for node_id, demand_dict in self.demand_data.items():
                if node_id not in nodes_to_fix:
                    total_demand = demand_dict.get('total', 0)
                    if not pd.isna(total_demand) and not np.isnan(total_demand) and total_demand > 0:
                        valid_demands.append(total_demand)
            
            avg_demand = np.mean(valid_demands) if valid_demands else expected_daily_demand / len(self.customer_nodes)
            
            # Fix problematic nodes
            for node_id in nodes_to_fix:
                self.demand_data[node_id] = {
                    'total': avg_demand,
                    'pickup': avg_demand * self.pickup_ratio,
                    'delivery': avg_demand * self.delivery_ratio
                }
        
        # Ensure all customer nodes have demand data
        missing_nodes = []
        for node_id in self.customer_nodes:
            if node_id not in self.demand_data:
                missing_nodes.append(node_id)
        
        if missing_nodes:
            print(f"Adding demand data for {len(missing_nodes)} missing nodes...")
            avg_demand = expected_daily_demand / len(self.customer_nodes)
            for node_id in missing_nodes:
                self.demand_data[node_id] = {
                    'total': avg_demand,
                    'pickup': avg_demand * self.pickup_ratio,
                    'delivery': avg_demand * self.delivery_ratio
                }
    
        # Final validation
        total_pickup_demand = sum(data.get('pickup', 0) for data in self.demand_data.values())
        total_delivery_demand = sum(data.get('delivery', 0) for data in self.demand_data.values())
        total_demand_check = total_pickup_demand + total_delivery_demand
        
        # Check for remaining NaN values
        if pd.isna(total_pickup_demand) or pd.isna(total_delivery_demand) or pd.isna(total_demand_check):
            print("ERROR: Still have NaN values after validation - using emergency fallback")
            # Emergency fallback to uniform distribution
            self._create_uniform_demand(expected_daily_demand)
            
            # Recalculate totals
            total_pickup_demand = sum(data.get('pickup', 0) for data in self.demand_data.values())
            total_delivery_demand = sum(data.get('delivery', 0) for data in self.demand_data.values())
            total_demand_check = total_pickup_demand + total_delivery_demand
        
        print(f"Total daily demand: {expected_daily_demand:.1f} parcels")
        print(f"Pickup demand: {total_pickup_demand:.1f} parcels ({total_pickup_demand/expected_daily_demand:.1%})")
        print(f"Delivery demand: {total_delivery_demand:.1f} parcels ({total_delivery_demand/expected_daily_demand:.1%})")
        print(f"Average demand per node: {np.mean([data['total'] for data in self.demand_data.values()]):.2f} parcels")
        print(f"Demand validation - Total calculated: {total_demand_check:.1f}, Expected: {expected_daily_demand:.1f}")
        
        # Final check
        if abs(total_demand_check - expected_daily_demand) / expected_daily_demand > 0.01:  # More than 1% difference
            print(f"Warning: Demand totals don't match expected values")

    def _load_existing_assignments(self, assignments_file: str):
        """Load existing customer-to-service-point assignments."""
        try:
            assignments_df = pd.read_csv(assignments_file)
            self.existing_assignments = {}
            
            for _, row in assignments_df.iterrows():
                customer_id = int(row['node_id'])
                service_point_id = int(row['closest_service_point'])
                distance_m = float(row['distance_m']);
                self.existing_assignments[customer_id] = {
                    'service_point': service_point_id,
                    'distance_km': distance_m / 1000
                }
            
            print(f"Loaded {len(self.existing_assignments)} existing assignments")
        except Exception as e:
            print(f"Could not load existing assignments: {e}")
            self.existing_assignments = {}

    def create_initial_solution(self, strategy: str = 'greedy') -> Dict:
        """Create initial solution using different strategies."""
        print(f"Creating initial solution using {strategy} strategy...")
        
        n_customers = len(self.customer_nodes)
        n_locations = len(self.candidate_locations)
        n_apls = len(self.apl_locations)
        
        # Initialize solution arrays
        y = np.zeros(n_locations, dtype=bool)  # Service points open
        apl_open = np.zeros(n_apls, dtype=bool)  # APLs open
        x_assigned = np.zeros((n_customers, n_locations + n_apls), dtype=bool)  # Customer to SP/APL assignments
        capacity = np.zeros(n_locations + n_apls, dtype=np.float32)  # Service point/APL capacities
        
        if strategy == 'existing' and hasattr(self, 'existing_assignments'):
            return self._create_existing_solution(y, apl_open, x_assigned, capacity)
        else:
            return self._create_random_solution(y, apl_open, x_assigned, capacity)
    
    def _create_existing_solution(self, y, apl_open, x_assigned, capacity):
        """Create solution based on existing assignments and force a set of random APLs open."""
        if not hasattr(self, 'existing_assignments') or not self.existing_assignments:
            return self._create_greedy_solution(y, x_assigned, capacity)
        
        # Get unique service points from existing assignments
        existing_sps = set(data['service_point'] for data in self.existing_assignments.values())
        
        # Open service points that are in our candidate locations
        for sp_id in existing_sps:
            if sp_id in self.candidate_locations:
                sp_idx = self.candidate_locations.index(sp_id)
                y[sp_idx] = True
        
        # Also open APLs that appear in the existing assignments
        for sp_id in existing_sps:
            if sp_id in self.apl_locations:
                apl_idx = self.apl_locations.index(sp_id)
                apl_open[apl_idx] = True
        
        # --- New block: Force a fixed number of random APLs open (e.g., 20) ---
        n_random_apls = 10
        current_open_apls = list(np.where(apl_open)[0])
        num_to_open = n_random_apls - len(current_open_apls)
        if num_to_open > 0:
            available_apl_indices = [i for i in range(len(self.apl_locations)) if i not in current_open_apls]
            random_apls = random.sample(available_apl_indices, num_to_open)
            for apl_idx in random_apls:
                apl_open[apl_idx] = True
        # ---------------------------------------------------------------------
        
        # Assign customers based on existing assignments
        pickup_demands = {}  # Track pickup demand for service points
        
        for cust_idx, cust_id in enumerate(self.customer_nodes):
            if cust_id in self.existing_assignments:
                assignment = self.existing_assignments[cust_id]
                sp_id = assignment['service_point']
                
                if sp_id in self.candidate_locations:
                    sp_idx = self.candidate_locations.index(sp_id)
                    if y[sp_idx]:
                        x_assigned[cust_idx, sp_idx] = True
                        pickup_demands[sp_idx] = pickup_demands.get(sp_idx, 0) + self.demand_data[cust_id]['pickup']
                        continue
                elif sp_id in self.apl_locations:
                    apl_idx = self.apl_locations.index(sp_id)
                    if apl_open[apl_idx]:
                        # Index for APLs in x_assigned is after candidate_locations
                        assign_idx = len(self.candidate_locations) + apl_idx
                        x_assigned[cust_idx, assign_idx] = True
                        continue
            
            # Fallback: assign customer to nearest open service point
            self._assign_customer_to_nearest({'y': y, 'x_assigned': x_assigned, 'capacity': capacity}, cust_idx)
        
        # Set capacities for service points based on pickup demand
        for sp_idx in range(len(self.candidate_locations)):
            if y[sp_idx] and sp_idx in pickup_demands:
                pickup_demand = pickup_demands[sp_idx]
                required_capacity = pickup_demand / (1 - self.parameters['BR_sp'] * 0.5)
                capacity[sp_idx] = min(required_capacity, self.parameters['M'])
        
        # Set fixed capacity for opened APLs
        for apl_idx in range(len(self.apl_locations)):
            assign_idx = len(self.candidate_locations) + apl_idx
            if apl_open[apl_idx]:
                capacity[assign_idx] = self.apl_capacity
        
        return {
            'y': y,
            'apl_open': apl_open,
            'x_assigned': x_assigned,
            'capacity': capacity
        }

    def _create_random_solution(self, y, apl_open, x_assigned, capacity):
            """Create random solution using a random initial solution.
        
            Randomly opens 20 service points and 30 APLs.
            APLs handle both pickup and delivery demand (50% each).
            Each customer is assigned to the nearest open facility (SP or APL).
            """
            n_customers = len(self.customer_nodes)
            n_locations = len(self.candidate_locations)
            n_apls = len(self.apl_locations)
            
            print(f"Creating random solution with {n_apls} available APLs")
            print(f"Target: 20 SPs and 30 APLs")
            
            # Open exactly 20 service points
            n_open_sp = 30
            open_locations = random.sample(range(n_locations), n_open_sp)
            for loc_idx in open_locations:
                y[loc_idx] = True
            
            # Open exactly 30 APLs (or all available if less than 30)
            n_open_apl = 20
            if n_apls > 0:
                open_apl_indices = random.sample(range(n_apls), n_open_apl)
                for apl_idx in open_apl_indices:
                    apl_open[apl_idx] = True
                print(f"Opened {len(open_apl_indices)} APLs at indices {open_apl_indices}")
                print(f"APL nodes opened: {[self.apl_locations[i] for i in open_apl_indices]}")
            
            # Assign each customer to the nearest open facility (SP or APL)
            sp_assignments = 0
            apl_assignments = 0
            
            for cust_idx in range(n_customers):
                cust_id = self.customer_nodes[cust_idx]
                cust_node_idx = self.node_to_idx[cust_id]
                best_distance = np.inf
                best_facility = None  # Tuple: (facility_index, facility_type); type is 'SP' or 'APL'
                
                # Check open service points
                for loc_idx in open_locations:
                    sp_id = self.candidate_locations[loc_idx]
                    sp_node_idx = self.node_to_idx[sp_id]
                    distance = self.distance_matrix[cust_node_idx, sp_node_idx]
                    if distance < best_distance and not np.isinf(distance):
                        best_distance = distance
                        best_facility = (loc_idx, 'SP')
                
                # Check open APLs
                if n_apls > 0:
                    for apl_idx in range(n_apls):
                        if apl_open[apl_idx]:
                            apl_id = self.apl_locations[apl_idx]
                            apl_node_idx = self.node_to_idx[apl_id]
                            distance = self.distance_matrix[cust_node_idx, apl_node_idx]
                            if distance < best_distance and not np.isinf(distance):
                                best_distance = distance
                                best_facility = (apl_idx, 'APL')
                
                # Assign customer based on best facility found
                if best_facility is not None:
                    if best_facility[1] == 'SP':
                        x_assigned[cust_idx, best_facility[0]] = True
                        sp_assignments += 1
                    else:
                        # APL columns start after candidate_locations columns
                        assign_idx = n_locations + best_facility[0]
                        x_assigned[cust_idx, assign_idx] = True
                        apl_assignments += 1
            
            print(f"Initial assignments: {sp_assignments} to SPs, {apl_assignments} to APLs")
            
            # Set capacities for open service points based on pickup demand only (SPs handle pickup)
            for loc_idx in open_locations:
                pickup_demand = sum(
                    self.demand_data[self.customer_nodes[cust_idx]]['pickup']
                    for cust_idx in range(n_customers)
                    if x_assigned[cust_idx, loc_idx]
                )
                if pickup_demand > 0:
                    required_capacity = pickup_demand / (1 - self.parameters['BR_sp'])
                    capacity[loc_idx] = min(required_capacity * 1.1, self.parameters['M'])
            
            # Set capacity for opened APLs to fixed value (60 parcels/day)
            for apl_idx in range(n_apls):
                assign_idx = n_locations + apl_idx
                if apl_open[apl_idx]:
                    capacity[assign_idx] = self.apl_capacity
            
            print(f"Created solution with {np.sum(y)} service points and {np.sum(apl_open)} APLs")
            
            return {
                'y': y,
                'apl_open': apl_open,
                'x_assigned': x_assigned,
                'capacity': capacity
            }

    def evaluate_solution(self, solution: Dict) -> float:
        """Evaluate the solution with detailed delivery cost calculations."""
        y = solution['y']
        x_assigned = solution['x_assigned']
        capacity = solution['capacity']
        apl_open = solution.get('apl_open', np.zeros(len(self.apl_locations), dtype=bool))
        
        # Pre-compute commonly used values
        open_locations = np.where(y)[0]
        n_customers = len(self.customer_nodes)
        n_candidate_locations = len(self.candidate_locations)
        
        # Fixed costs
        fixed_costs = np.sum(y) * self.parameters['F_SP']
        apl_fixed_costs = np.sum(apl_open) * self.apl_fixed_cost
        capacity_costs = np.sum(capacity) * self.parameters['C_cap']
        
        # Delivery cost calculation (for both SPs and APLs)
        delivery_costs = 0
        customer_assignments = np.where(x_assigned)  # Get (customer_idx, assignment_idx) pairs
    
        for cust_idx, assign_idx in zip(customer_assignments[0], customer_assignments[1]):
            cust_id = self.customer_nodes[cust_idx]
            
            # Determine if this is a service point or APL assignment
            if assign_idx < n_candidate_locations:
                # Service point assignment - only delivery cost
                loc_id = self.candidate_locations[assign_idx]
                delivery_demand = self.demand_data[cust_id]['delivery']
            else:
                # APL assignment - both pickup and delivery cost (APLs handle both)
                apl_idx = assign_idx - n_candidate_locations
                loc_id = self.apl_locations[apl_idx]
                # APLs handle both pickup and delivery, so use total demand
                delivery_demand = self.demand_data[cust_id]['delivery'] + self.demand_data[cust_id]['pickup']
            
            # Calculate delivery cost
            distance = self.distance_matrix[self.node_to_idx[cust_id], self.node_to_idx[loc_id]]
            delivery_costs += delivery_demand * distance * self.parameters['C_hd'] * 365
        
        # Constraint checking
        penalty = 0
        penalty_weight = 1e6
        
        # Check assignment constraints (each customer assigned exactly once)
        assignment_counts = np.sum(x_assigned, axis=1)
        penalty += np.sum(np.abs(assignment_counts - 1)) * penalty_weight
        
        # Check assignments to closed service points
        for loc_idx in range(n_candidate_locations):
            if not y[loc_idx]:  # Service point is closed
                closed_assignments = np.sum(x_assigned[:, loc_idx])
                penalty += closed_assignments * penalty_weight
        
        # Check assignments to closed APLs
        for apl_idx in range(len(self.apl_locations)):
            assign_idx = n_candidate_locations + apl_idx
            if not apl_open[apl_idx]:  # APL is closed
                closed_assignments = np.sum(x_assigned[:, assign_idx])
                penalty += closed_assignments * penalty_weight
        
        # Bounce rate calculation for service points only (APLs have different handling)
        pickup_demands = np.zeros(n_candidate_locations)
        for cust_idx in range(n_customers):
            for loc_idx in range(n_candidate_locations):
                if x_assigned[cust_idx, loc_idx]:
                    cust_id = self.customer_nodes[cust_idx]
                    pickup_demands[loc_idx] += self.demand_data[cust_id]['pickup']
        
        # Bounce rate penalties for service points
        valid_locations = pickup_demands > 0
        if np.any(valid_locations):
            bounces = np.maximum(0, pickup_demands - capacity[:n_candidate_locations])
            sp_bounce_rates = bounces / np.maximum(pickup_demands, 1e-8)
            
            # Per-service point penalties
            excessive_bounce = sp_bounce_rates > self.parameters['BR_sp']
            penalty += np.sum(
                (sp_bounce_rates[excessive_bounce] - self.parameters['BR_sp']) * 
                pickup_demands[excessive_bounce] * penalty_weight
            )
            
            # City-wide bounce rate (service points only)
            total_pickup = np.sum(pickup_demands)
            total_bounced = np.sum(bounces)
            if total_pickup > 0:
                city_bounce_rate = total_bounced / total_pickup
                if city_bounce_rate > self.parameters['BR_cw']:
                    penalty += (city_bounce_rate - self.parameters['BR_cw']) * total_pickup * penalty_weight
        
        # APL capacity check (APLs handle both pickup and delivery)
        for apl_idx in range(len(self.apl_locations)):
            assign_idx = n_candidate_locations + apl_idx
            if apl_open[apl_idx]:
                total_apl_demand = 0
                for cust_idx in range(n_customers):
                    if x_assigned[cust_idx, assign_idx]:
                        cust_id = self.customer_nodes[cust_idx]
                        total_apl_demand += self.demand_data[cust_id]['pickup'] + self.demand_data[cust_id]['delivery']
                
                # Check if APL is over capacity
                if total_apl_demand > capacity[assign_idx]:
                    penalty += (total_apl_demand - capacity[assign_idx]) * penalty_weight
    
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

        # Add delivery costs to total cost
        total_cost = fixed_costs + apl_fixed_costs + capacity_costs + delivery_costs + penalty

        return total_cost

    def generate_neighbor(self, solution: Dict) -> Dict:
        """Generate neighbor solution with capacity adjustment."""
        # Create deep copy
        neighbor = {
            'y': solution['y'].copy(),
            'apl_open': solution.get('apl_open', np.zeros(len(self.apl_locations), dtype=bool)).copy(),
            'x_assigned': solution['x_assigned'].copy(),
            'capacity': solution['capacity'].copy()
        }
        
        # Choose move type - now includes APL operations
        move_types = [
            ('toggle_service_point', 0.3),
            ('reassign_customer', 0.2),
            ('adjust_capacity', 0.15),
            ('relocate_service_point', 0.2),
            ('toggle_apl', 0.05),
            ('relocate_apl', 0.1)
        ]
        
        # Weighted random selection
        rand = random.random()
        cumulative = 0
        for move_type, weight in move_types:
            cumulative += weight
            if rand <= cumulative:
                break
        
        if move_type == 'toggle_service_point':
            self._toggle_service_point(neighbor)
        elif move_type == 'reassign_customer':
            self._reassign_customer(neighbor)
        elif move_type == 'adjust_capacity':
            self._adjust_capacity(neighbor)
        elif move_type == 'relocate_service_point':
            self._relocate_service_point(neighbor)
        elif move_type == 'toggle_apl':
            self._toggle_apl(neighbor)
        elif move_type == 'relocate_apl':
            self._relocate_apl(neighbor)
        
        # Always adjust capacities after making changes
        self._adjust_capacity_after_assignment(neighbor)
        
        return neighbor

    def _toggle_apl(self, solution: Dict):
        """Toggle an APL open/closed."""
        if len(self.apl_locations) == 0:
            return
        
        apl_open = solution['apl_open']
        open_apls = np.where(apl_open)[0]
        closed_apls = np.where(~apl_open)[0]
        
        if random.random() < 0.5 and len(open_apls) > 0:
            # Close an APL
            apl_to_close = random.choice(open_apls)
            apl_open[apl_to_close] = False
            
            # Clear capacity for this APL
            apl_assign_idx = len(self.candidate_locations) + apl_to_close
            solution['capacity'][apl_assign_idx] = 0
            
            # Reassign customers from this APL
            affected_customers = np.where(solution['x_assigned'][:, apl_assign_idx])[0]
            
            for cust_idx in affected_customers:
                solution['x_assigned'][cust_idx, apl_assign_idx] = False
                self._assign_customer_to_nearest_facility(solution, cust_idx)
        else:
            # Open an APL
            if len(closed_apls) > 0:
                apl_to_open = random.choice(closed_apls)
                apl_open[apl_to_open] = True
                
                # Set APL capacity
                apl_assign_idx = len(self.candidate_locations) + apl_to_open
                solution['capacity'][apl_assign_idx] = self.apl_capacity

    def _relocate_apl(self, solution: Dict):
        """Move an APL from one location to another."""
        if len(self.apl_locations) == 0:
            return
        
        # Find current open APLs
        apl_open = solution['apl_open']
        open_apls = np.where(apl_open)[0]
        
        if len(open_apls) == 0:
            return
        
        # Choose an APL to relocate
        apl_to_relocate = random.choice(open_apls)
        old_apl_id = self.apl_locations[apl_to_relocate]
        
        # Find potential new locations (from candidate locations not currently used as APLs)
        current_apl_locations_set = set(self.apl_locations)
        potential_new_locations = [loc for loc in self.candidate_locations 
                                 if loc not in current_apl_locations_set]
        
        if not potential_new_locations:
            return
        
        # Choose new location
        new_apl_id = random.choice(potential_new_locations)
        
        # Update APL locations list
        old_assign_idx = len(self.candidate_locations) + apl_to_relocate
        
        # Clear assignments from old location
        affected_customers = np.where(solution['x_assigned'][:, old_assign_idx])[0]
        solution['x_assigned'][:, old_assign_idx] = False
        solution['capacity'][old_assign_idx] = 0
        
        # Update the APL location
        self.apl_locations[apl_to_relocate] = new_apl_id
        
        # Set capacity for new location
        solution['capacity'][old_assign_idx] = self.apl_capacity
        
        # Reassign affected customers to nearest facilities
        for cust_idx in affected_customers:
            self._assign_customer_to_nearest_facility(solution, cust_idx)
        
    def _assign_customer_to_nearest_facility(self, solution: Dict, cust_idx: int):
        """Assign customer to nearest open facility (service point or APL)."""
        cust_id = self.customer_nodes[cust_idx]
        cust_node_idx = self.node_to_idx[cust_id]
        
        best_distance = np.inf
        best_facility = None
        best_facility_type = None
        
        # Check open service points
        open_sp = np.where(solution['y'])[0]
        for sp_idx in open_sp:
            sp_id = self.candidate_locations[sp_idx]
            sp_node_idx = self.node_to_idx[sp_id]
            distance = self.distance_matrix[cust_node_idx, sp_node_idx]
            
            if distance < best_distance and not np.isinf(distance):
                best_distance = distance
                best_facility = sp_idx
                best_facility_type = 'SP'
        
        # Check open APLs
        if len(self.apl_locations) > 0:
            apl_open = solution.get('apl_open', np.zeros(len(self.apl_locations), dtype=bool))
            open_apls = np.where(apl_open)[0]
            
            for apl_idx in open_apls:
                apl_id = self.apl_locations[apl_idx]
                apl_node_idx = self.node_to_idx[apl_id]
                distance = self.distance_matrix[cust_node_idx, apl_node_idx]
                
                if distance < best_distance and not np.isinf(distance):
                    best_distance = distance
                    best_facility = apl_idx
                    best_facility_type = 'APL'
        
        # Assign to best facility found
        if best_facility is not None:
            if best_facility_type == 'SP':
                solution['x_assigned'][cust_idx, best_facility] = True
            else:  # APL
                apl_assign_idx = len(self.candidate_locations) + best_facility
                solution['x_assigned'][cust_idx, apl_assign_idx] = True
        else:
            print(f"Warning: Could not assign customer {cust_id} to any facility")

    def _toggle_service_point(self, solution: Dict):
        """Toggle a service point open/closed."""
        y = solution['y']
        open_sp = np.where(y)[0]
        closed_sp = np.where(~y)[0]
        
        if random.random() < 0.5 and len(open_sp) > 1:
            # Close a service point
            sp_to_close = random.choice(open_sp)
            y[sp_to_close] = False
            solution['capacity'][sp_to_close] = 0
            
            # Reassign customers
            affected_customers = np.where(solution['x_assigned'][:, sp_to_close])[0]
            
            for cust_idx in affected_customers:
                solution['x_assigned'][cust_idx, sp_to_close] = False
                self._assign_customer_to_nearest(solution, cust_idx)
        else:
            # Open a service point
            if len(closed_sp) > 0:
                sp_to_open = random.choice(closed_sp)
                y[sp_to_open] = True
                solution['capacity'][sp_to_open] = 100  # Will be adjusted later

    def _reassign_customer(self, solution: Dict):
        """Reassign a customer to a different service point."""
        cust_idx = random.randint(0, len(self.customer_nodes) - 1)
        
        # Clear current assignment
        solution['x_assigned'][cust_idx, :] = False
        
        # Assign to new service point
        self._assign_customer_to_nearest(solution, cust_idx)

    def _adjust_capacity(self, solution: Dict):
        """Adjust capacity of a random open service point."""
        open_sp = np.where(solution['y'])[0]
        if len(open_sp) > 0:
            sp_idx = random.choice(open_sp)
            current_capacity = solution['capacity'][sp_idx]
            
            # Adjust by ±20%
            adjustment = random.uniform(0.8, 1.2)
            new_capacity = min(current_capacity * adjustment, self.parameters['M'])
            solution['capacity'][sp_idx] = max(new_capacity, 1.0)

    def _relocate_service_point(self, solution: Dict):
        """Close one service point and open another nearby."""
        open_sp = np.where(solution['y'])[0]
        closed_sp = np.where(~solution['y'])[0]
        
        if len(open_sp) > 1 and len(closed_sp) > 0:
            # Close one
            sp_to_close = random.choice(open_sp)
            solution['y'][sp_to_close] = False
            solution['capacity'][sp_to_close] = 0
            
            # Open nearby one
            close_loc_id = self.candidate_locations[sp_to_close]
            close_node_idx = self.node_to_idx[close_loc_id]
            
            # Find nearby closed service points
            nearby_sp = []
            for sp_idx in closed_sp:
                sp_loc_id = self.candidate_locations[sp_idx]
                sp_node_idx = self.node_to_idx[sp_loc_id]
                distance = self.distance_matrix[close_node_idx, sp_node_idx]
                if distance <= 2.0:  # Within 2km
                    nearby_sp.append(sp_idx)
            
            if nearby_sp:
                sp_to_open = random.choice(nearby_sp)
                solution['y'][sp_to_open] = True
                solution['capacity'][sp_to_open] = 100
            
            # Reassign affected customers
            affected_customers = np.where(solution['x_assigned'][:, sp_to_close])[0]
            
            for cust_idx in affected_customers:
                solution['x_assigned'][cust_idx, sp_to_close] = False
                self._assign_customer_to_nearest(solution, cust_idx)

    def _assign_customer_to_nearest(self, solution: Dict, cust_idx: int):
        """Assign customer to nearest open facility (service point or APL) with fallback for unreachable locations."""
        cust_id = self.customer_nodes[cust_idx]
        cust_node_idx = self.node_to_idx[cust_id]
        
        best_distance = np.inf
        best_facility = None
        best_facility_type = None
        
        # Check open service points
        open_sp = np.where(solution['y'])[0]
        for sp_idx in open_sp:
            sp_id = self.candidate_locations[sp_idx]
            sp_node_idx = self.node_to_idx[sp_id]
            distance = self.distance_matrix[cust_node_idx, sp_node_idx]
            
            if distance < best_distance and not np.isinf(distance):
                best_distance = distance
                best_facility = sp_idx
                best_facility_type = 'SP'
        
        # Check open APLs
        if len(self.apl_locations) > 0:
            apl_open = solution.get('apl_open', np.zeros(len(self.apl_locations), dtype=bool))
            open_apls = np.where(apl_open)[0]
            
            for apl_idx in open_apls:
                apl_id = self.apl_locations[apl_idx]
                apl_node_idx = self.node_to_idx[apl_id]
                distance = self.distance_matrix[cust_node_idx, apl_node_idx]
                
                if distance < best_distance and not np.isinf(distance):
                    best_distance = distance
                    best_facility = apl_idx
                    best_facility_type = 'APL'
        
        # Assign to best facility found
        if best_facility is not None:
            if best_facility_type == 'SP':
                solution['x_assigned'][cust_idx, best_facility] = True
            else:  # APL
                apl_assign_idx = len(self.candidate_locations) + best_facility
                solution['x_assigned'][cust_idx, apl_assign_idx] = True
        else:
            # Fallback: if all distances are infinite, assign to first open facility
            if len(open_sp) > 0:
                print(f"Warning: Customer {cust_id} has no reachable facilities, assigning to first available SP")
                solution['x_assigned'][cust_idx, open_sp[0]] = True
            else:
                print(f"ERROR: Could not assign customer {cust_id} to any facility")

    def _adjust_capacity_after_assignment(self, solution: Dict):
        """Adjust service point capacities after customer assignments."""
        y = solution['y']
        x_assigned = solution['x_assigned']
        capacity = solution['capacity']
        
        for loc_idx in range(len(self.candidate_locations)):
            if y[loc_idx]:
                # Calculate total pickup demand for this service point
                pickup_demand = 0
                for cust_idx in range(len(self.customer_nodes)):
                    if x_assigned[cust_idx, loc_idx]:
                        cust_id = self.customer_nodes[cust_idx]
                        pickup_demand += self.demand_data[cust_id]['pickup']
                
                if pickup_demand > 0:
                    # Set capacity to handle demand with acceptable bounce rate
                    target_bounce_rate = self.parameters['BR_sp'] * 0.75  # Safety margin
                    required_capacity = pickup_demand / (1 - target_bounce_rate)
                    capacity[loc_idx] = min(required_capacity, self.parameters['M'])
                else:
                    capacity[loc_idx] = 0

    def optimize(self, initial_temp: float = 10000, cooling_rate: float = 0.995, 
                max_iterations: int = 100000, time_limit: int = 3600,
                initial_strategy: str = 'greedy') -> bool:
        """Run simulated annealing optimization."""
        print(f"Starting optimization with {initial_strategy} initial solution...")
        start_time = time.time()
        
        # Create initial solution
        self.current_solution = self.create_initial_solution(initial_strategy)
        current_cost = self.evaluate_solution(self.current_solution)
        
        self.best_solution = {k: v.copy() for k, v in self.current_solution.items()}
        self.best_cost = current_cost
        self.cost_history = [current_cost]
        
        print(f"Initial solution cost: €{current_cost:,.0f}")
        
        # Optimization parameters
        temperature = initial_temp
        iteration = 0
        accepted_moves = 0
        last_improvement = 0
        stagnation_limit = max_iterations 
        
        while (time.time() - start_time < time_limit and 
               iteration < max_iterations and
               iteration - last_improvement < stagnation_limit):
            
            # Generate neighbor
            neighbor = self.generate_neighbor(self.current_solution)
            neighbor_cost = self.evaluate_solution(neighbor)
            
            # Acceptance decision
            cost_diff = neighbor_cost - current_cost
            if cost_diff <= 0 or random.random() < math.exp(-cost_diff / temperature):
                self.current_solution = neighbor
                current_cost = neighbor_cost
                accepted_moves += 1
                
                # Update best solution
                if neighbor_cost < self.best_cost:
                    self.best_solution = {k: v.copy() for k, v in neighbor.items()}
                    self.best_cost = neighbor_cost
                    last_improvement = iteration
                    print(f"New best solution: €{self.best_cost:,.0f} (iteration {iteration})")
            
            # Cool down
            temperature *= cooling_rate
            iteration += 1
           
            # Track progress
            if iteration % 1000 == 0:
                self.cost_history.append(current_cost)
                elapsed_time = time.time() - start_time
                acceptance_rate = accepted_moves / iteration if iteration > 0 else 0
                
                print(f"Iteration {iteration:,}: Cost=€{current_cost:,.0f}, "
                      f"Best=€{self.best_cost:,.0f}, Temp={temperature:.1f}, "
                      f"Accept={acceptance_rate:.1%}, Time={elapsed_time:.1f}s")
        
        # Final optimization statistics
        elapsed_time = time.time() - start_time
        acceptance_rate = accepted_moves / iteration if iteration > 0 else 0
        
        print(f"\nOptimization completed!")
        print(f"Total iterations: {iteration:,}")
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Final acceptance rate: {acceptance_rate:.1%}")
        print(f"Best solution cost: €{self.best_cost:,.0f}")
        
        # Validate final solution
        self._validate_solution(self.best_solution)
        
        return True

    def _validate_solution(self, solution: Dict):
        """Validate solution constraints and print statistics."""
        y = solution['y']
        x_assigned = solution['x_assigned']
        capacity = solution['capacity']
        apl_open = solution.get('apl_open', np.zeros(len(self.apl_locations), dtype=bool))
        
        n_service_points = np.sum(y)
        n_apls = np.sum(apl_open)
        total_capacity = np.sum(capacity)
        
        print(f"\n=== Solution Validation ===")
        print(f"Number of service points: {n_service_points}")
        print(f"Number of APLs: {n_apls}")
        print(f"Total capacity: {total_capacity:,.0f} parcels/day")
        
        # Check customer assignments
        unassigned_customers = 0
        multi_assigned_customers = 0
        sp_assignments = 0
        apl_assignments = 0
        
        for cust_idx in range(len(self.customer_nodes)):
            assignments = np.sum(x_assigned[cust_idx])
            if assignments == 0:
                unassigned_customers += 1
            elif assignments > 1:
                multi_assigned_customers += 1
            else:
                # Count assignment type
                sp_assigned = np.sum(x_assigned[cust_idx, :len(self.candidate_locations)])
                if sp_assigned > 0:
                    sp_assignments += 1
                else:
                    apl_assignments += 1
        
        print(f"Unassigned customers: {unassigned_customers}")
        print(f"Multi-assigned customers: {multi_assigned_customers}")
        print(f"Customers assigned to service points: {sp_assignments}")
        print(f"Customers assigned to APLs: {apl_assignments}")
        print(f"Total assigned customers: {sp_assignments + apl_assignments}")
        
        # Calculate total customers with pickup demand
        total_customers_with_pickup = len([cust_id for cust_id in self.customer_nodes 
                                         if self.demand_data[cust_id]['pickup'] > 0])
        total_customers = len(self.customer_nodes)
        
        print(f"Total customers in network: {total_customers}")
        print(f"Total customers with pickup demand: {total_customers_with_pickup}")
        
        # Verify assignment coverage
        assignment_coverage = (sp_assignments + apl_assignments) / total_customers_with_pickup if total_customers_with_pickup > 0 else 0
        print(f"Assignment coverage: {assignment_coverage:.1%} of customers with pickup demand")
        
        if sp_assignments + apl_assignments != total_customers_with_pickup:
            print(f"WARNING: Assignment mismatch! Assigned: {sp_assignments + apl_assignments}, Expected: {total_customers_with_pickup}")
        else:
            print("✓ All customers with pickup demand are properly assigned")
        
        # Calculate pickup demand verification
        total_pickup_demand_theoretical = sum(data['pickup'] for data in self.demand_data.values())
        
        # Calculate actual pickup demand being handled
        pickup_demand_handled_by_sps = 0
        pickup_demand_handled_by_apls = 0
        
        for cust_idx in range(len(self.customer_nodes)):
            cust_id = self.customer_nodes[cust_idx]
            customer_pickup_demand = self.demand_data[cust_id]['pickup']
            
            # Check SP assignments
            for loc_idx in range(len(self.candidate_locations)):
                if x_assigned[cust_idx, loc_idx]:
                    pickup_demand_handled_by_sps += customer_pickup_demand
                    break
            
            # Check APL assignments
            for apl_idx in range(len(self.apl_locations)):
                assign_idx = len(self.candidate_locations) + apl_idx
                if x_assigned[cust_idx, assign_idx]:
                    pickup_demand_handled_by_apls += customer_pickup_demand
                    break
        
        total_pickup_demand_handled = pickup_demand_handled_by_sps + pickup_demand_handled_by_apls
        
        print(f"\n=== Pickup Demand Verification ===")
        print(f"Theoretical total pickup demand: {total_pickup_demand_theoretical:.1f} parcels/day")
        print(f"Pickup demand handled by SPs: {pickup_demand_handled_by_sps:.1f} parcels/day")
        print(f"Pickup demand handled by APLs: {pickup_demand_handled_by_apls:.1f} parcels/day")
        print(f"Total pickup demand handled: {total_pickup_demand_handled:.1f} parcels/day")
        
        demand_coverage = total_pickup_demand_handled / total_pickup_demand_theoretical if total_pickup_demand_theoretical > 0 else 0
        print(f"Pickup demand coverage: {demand_coverage:.1%}")
        
        if abs(total_pickup_demand_handled - total_pickup_demand_theoretical) > 0.1:
            print(f"WARNING: Pickup demand mismatch! Handled: {total_pickup_demand_handled:.1f}, Expected: {total_pickup_demand_theoretical:.1f}")
        else:
            print("✓ All pickup demand is properly covered")
        
        # Continue with bounce rate calculations...
        total_pickup_demand = 0
        total_bounced = 0
        max_sp_bounce_rate = 0
        
        for loc_idx in range(len(self.candidate_locations)):
            if y[loc_idx]:
                pickup_demand = 0
                for cust_idx in range(len(self.customer_nodes)):
                    if x_assigned[cust_idx, loc_idx]:
                        cust_id = self.customer_nodes[cust_idx]
                        pickup_demand += self.demand_data[cust_id]['pickup']
                
                if pickup_demand > 0:
                    total_pickup_demand += pickup_demand
                    bounce = max(0, pickup_demand - capacity[loc_idx])
                    total_bounced += bounce
                    
                    sp_bounce_rate = bounce / pickup_demand
                    max_sp_bounce_rate = max(max_sp_bounce_rate, sp_bounce_rate)
        
        city_bounce_rate = total_bounced / total_pickup_demand if total_pickup_demand > 0 else 0
        
        print(f"\n=== Bounce Rate Analysis ===")
        print(f"Service point pickup demand: {total_pickup_demand:.1f} parcels/day")
        print(f"City-wide bounce rate: {city_bounce_rate:.1%} (limit: {self.parameters['BR_cw']:.1%})")
        print(f"Max service point bounce rate: {max_sp_bounce_rate:.1%} (limit: {self.parameters['BR_sp']:.1%})")
        
        # Rest of the cost breakdown remains the same...
        fixed_costs = n_service_points * self.parameters['F_SP']
        apl_fixed_costs = n_apls * self.apl_fixed_cost
        capacity_costs = total_capacity * self.parameters['C_cap']
        
        delivery_costs = 0
        total_delivery_distance = 0
        total_customers_served = 0
        
        # Calculate delivery costs for both SPs and APLs
        for cust_idx in range(len(self.customer_nodes)):
            cust_id = self.customer_nodes[cust_idx]
            cust_node_idx = self.node_to_idx[cust_id]
            
            # Check service point assignments
            for loc_idx in range(len(self.candidate_locations)):
                if x_assigned[cust_idx, loc_idx]:
                    loc_id = self.candidate_locations[loc_idx]
                    loc_node_idx = self.node_to_idx[loc_id]
                    distance = self.distance_matrix[cust_node_idx, loc_node_idx]
                    delivery_demand = self.demand_data[cust_id]['delivery']
                    delivery_costs += delivery_demand * distance * self.parameters['C_hd'] * 365
                    total_delivery_distance += distance
                    total_customers_served += 1
                    break
            
            # Check APL assignments
            for apl_idx in range(len(self.apl_locations)):
                assign_idx = len(self.candidate_locations) + apl_idx
                if x_assigned[cust_idx, assign_idx]:
                    apl_id = self.apl_locations[apl_idx]
                    apl_node_idx = self.node_to_idx[apl_id]
                    distance = self.distance_matrix[cust_node_idx, apl_node_idx]
                    delivery_demand = self.demand_data[cust_id]['delivery']
                    delivery_costs += delivery_demand * distance * self.parameters['C_hd'] * 365
                    total_delivery_distance += distance
                    total_customers_served += 1
                    break
        
        avg_distance = total_delivery_distance / max(1, total_customers_served)
        
        print(f"\n=== Cost Breakdown ===")
        print(f"Fixed costs (SPs): €{fixed_costs:,.0f}")
        print(f"Fixed costs (APLs): €{apl_fixed_costs:,.0f}")
        print(f"Capacity costs: €{capacity_costs:,.0f}")
        print(f"Delivery costs: €{delivery_costs:,.0f}")
        print(f"Total cost: €{fixed_costs + apl_fixed_costs + capacity_costs + delivery_costs:,.0f}")
        
        print(f"\n=== Service Statistics ===")
        print(f"Total customers served: {total_customers_served:,}")
        print(f"Average distance to assigned facility: {avg_distance:.2f} km")
        print(f"Pickup demand ratio: {self.pickup_ratio:.1%}")
        print(f"Delivery demand ratio: {self.delivery_ratio:.1%}")
    
    def _validate_data(self):
        """Validate that all required data is properly loaded and contains no NaN values."""
        print("Validating loaded data...")
        
        # Check demand data
        if not self.demand_data:
            raise ValueError("No demand data loaded")
        
        # Check for NaN values in demand data
        problems_found = False
        for node_id, demand in self.demand_data.items():
            for key, value in demand.items():
                if pd.isna(value) or np.isnan(value) or np.isinf(value):
                    print(f"ERROR: Found invalid value {value} for node {node_id}, key {key}")
                    problems_found = True
    
        if problems_found:
            raise ValueError("Invalid values found in demand data after processing")
        
        # Check distance matrix
        if np.all(np.isinf(self.distance_matrix)):
            raise ValueError("Distance matrix contains only infinite values")
        
        # Check for missing customer nodes in demand data
        missing_demand = [node for node in self.customer_nodes if node not in self.demand_data]
        if missing_demand:
            raise ValueError(f"Missing demand data for {len(missing_demand)} customer nodes")
        
        print("Data validation completed successfully")

    
    def visualize_solution_static(self, solution: Dict, save_path: str = None):
        import matplotlib.pyplot as plt
    
        print("Creating static solution visualization...")
    
        plt.figure(figsize=(18, 14))
    
        # 1) all nodes in light grey
        all_x = self.nodes_data['X'].values
        all_y = self.nodes_data['Y'].values
        plt.scatter(all_x, all_y, c='lightgrey', s=8, alpha=0.6, label='All Nodes', zorder=1)
    
        # 2) APLs in blue (no labels)
        opened_apls = [loc for i, loc in enumerate(solution.get('apl_open', [])) if loc and solution['apl_open'][i]]
        if opened_apls:
            apl_df = self.nodes_data[self.nodes_data['NODE ID'].isin(opened_apls)]
            missing = set(opened_apls) - set(apl_df['NODE ID'])
            if missing:
                print(f"Warning: could not find coords for {len(missing)} APLs: {sorted(missing)}")
            plt.scatter(apl_df['X'], apl_df['Y'],
                        c='blue', s=80, alpha=0.8,
                        label=f'APLs ({len(opened_apls)})', zorder=3)
    
        # 3) SPs in red with ID\nCap labels
        opened_sps = [nid for i, nid in enumerate(self.candidate_locations) if solution['y'][i]]
        sp_caps = {nid: solution['capacity'][i] for i, nid in enumerate(self.candidate_locations) if solution['y'][i]}
        if opened_sps:
            sp_df = self.nodes_data[self.nodes_data['NODE ID'].isin(opened_sps)]
            plt.scatter(sp_df['X'], sp_df['Y'],
                        c='red', s=100, alpha=0.9,
                        label=f'SPs ({len(opened_sps)})', zorder=4)
            for x, y, nid in zip(sp_df['X'], sp_df['Y'], sp_df['NODE ID']):
                cap = sp_caps.get(nid, 0)
                plt.annotate(f"{int(nid)}\nCap:{cap:.0f}",
                             (x, y), xytext=(5,5), textcoords='offset points',
                             ha='left', va='bottom',
                             fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', alpha=0.8),
                             zorder=5)
    
        # finalize
        total_cap = sum(sp_caps.values())
        plt.title(
            f"Service‐Point Network Solution\n"
            f"{len(opened_sps)} SPs, {len(opened_apls)} APLs, "
            f"TotCap: {total_cap:,.0f} parcels/day",
            fontsize=14, fontweight='bold'
        )
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
    
        plt.show()

    def _ensure_symmetric_distance_matrix_vectorized(self):
        """Vectorized version for large matrices."""
        print("Ensuring distance matrix symmetry (vectorized)...")
    
        # Get upper and lower triangular parts
        upper_tri = np.triu(self.distance_matrix, k=1)
        lower_tri = np.tril(self.distance_matrix, k=-1)
    
        # Find where only one direction exists
        upper_exists = ~np.isinf(upper_tri) & (upper_tri != 0)
        lower_exists = ~np.isinf(lower_tri.T) & (lower_tri.T != 0)
    
        # Take minimum where both exist, otherwise take the existing value
        symmetric_part = np.where(
            upper_exists & lower_exists,
            np.minimum(upper_tri, lower_tri.T),  # Both exist - take minimum
            np.where(upper_exists, upper_tri, lower_tri.T)  # Take whichever exists
        )
    
        # Apply symmetry
        self.distance_matrix = symmetric_part + symmetric_part.T
        np.fill_diagonal(self.distance_matrix, 0.0)
    
        print("Symmetry enforcement completed")

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

if __name__ == "__main__":
    """Main optimization workflow."""
    print("=== Enhanced Service Point Optimization ===")
    
    try:
        # Initialize optimizer
        optimizer = EnhancedServicePointOptimizer()
        
        # Load data
        print("\n1. Loading data...")
        optimizer.load_data(
            excel_file='data_Maastricht_2024.xlsx',
            distance_file='distance_matrix.csv',
            assignments_file='node_to_service_point_assignments_max5kmh.csv'  # Optional
        )
        
        print(f"\nData loaded successfully:")
        print(f"- {len(optimizer.customer_nodes):,} customer nodes")
        print(f"- {len(optimizer.candidate_locations)} candidate service point locations")
        print(f"- Distance matrix: {optimizer.distance_matrix.shape}")
        
        # Run optimization
        print("\n2. Starting optimization...")
        success = optimizer.optimize(
            initial_temp=100,        # Initial temperature for simulated annealing
            cooling_rate=0.995,        # Cooling rate
            max_iterations=1000,     # Maximum iterations
            time_limit=600,           # Time limit in seconds (30 minutes)
            initial_strategy='existing'  # Initial solution strategy
        )
        
        if success:
            print("\nOptimization completed successfully!")
            print(f"Best solution cost: €{optimizer.best_cost:,.0f}")
            
            # Validate and visualize the solution
            optimizer._validate_solution(optimizer.best_solution)
            optimizer.visualize_solution_static(optimizer.best_solution, save_path='service_points_solution.png')
            
            # Save results
            try:
                import pickle
                with open('optimization_results.pkl', 'wb') as f:
                    pickle.dump({
                        'best_solution': optimizer.best_solution,
                        'best_cost': optimizer.best_cost,   # continue
                        'parameters': optimizer.parameters
                    }, f)
                print("\nResults saved to 'optimization_results.pkl'")
            except Exception as e:
                print(f"Could not save results: {e}")
        else:
            print("Optimization failed!")
    
    except FileNotFoundError as e:
        print(f"\nError: Required file not found - {e}")
        print("Please ensure the following files are in the working directory:")
        print("- data_Maastricht_2024.xlsx")
        print("- distance_matrix.csv")
        print("- node_to_service_point_assignments_max5kmh.csv (optional)")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nService Point Optimizer completed!")
