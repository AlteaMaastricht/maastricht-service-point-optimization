import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import copy
import time
from SA5 import EnhancedServicePointOptimizer

class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for Post&L network optimization.
    Tests parameter variations and network robustness.
    """
    
    def __init__(self, base_optimizer: EnhancedServicePointOptimizer):
        """Initialize with a base optimizer that has data already loaded."""
        self.base_optimizer = base_optimizer
        self.base_parameters = copy.deepcopy(base_optimizer.parameters)
        self.base_apl_params = {
            'apl_fixed_cost': base_optimizer.apl_fixed_cost,
            'apl_capacity': base_optimizer.apl_capacity
        }
        self.base_demand_split = {
            'pickup_ratio': base_optimizer.pickup_ratio,
            'delivery_ratio': base_optimizer.delivery_ratio
        }
        
        # Results storage
        self.sensitivity_results = {}
        self.robustness_results = {}
        
    def run_parameter_sensitivity_analysis(self, 
                                         parameter_variations: Dict[str, List[float]] = None,
                                         max_iterations: int = 5000,
                                         time_limit: int = 600) -> Dict:
        """
        Test impact of parameter variations on optimal solutions.
        
        Args:
            parameter_variations: Dict of parameter names and their test values
            max_iterations: Max iterations per test
            time_limit: Time limit per test in seconds
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        print("=== PARAMETER SENSITIVITY ANALYSIS ===")
        
        # Default parameter variations if not provided
        if parameter_variations is None:
            parameter_variations = {
                'F_SP': [25000, 37500, 50000, 62500, 75000],  # ±50% around base (50,000)
                'C_hd': [0.75, 1.125, 1.50, 1.875, 2.25],    # ±50% around base (1.50)
                'BR_sp': [0.01, 0.015, 0.02, 0.025, 0.03],   # ±50% around base (0.02)
                'BR_cw': [0.005, 0.0075, 0.01, 0.0125, 0.015], # ±50% around base (0.01)
                'apl_fixed_cost': [250, 375, 500, 625, 750],  # ±50% around base (500)
                'pickup_ratio': [0.4, 0.485, 0.57, 0.655, 0.74] # ±30% around base (0.57)
            }
        
        # Store baseline result first
        print("\n1. Running baseline optimization...")
        baseline_result = self._run_single_optimization(
            "baseline", {}, max_iterations, time_limit
        )
        
        sensitivity_results = {
            'baseline': baseline_result,
            'parameter_tests': {}
        }
        
        # Test each parameter
        for param_name, test_values in parameter_variations.items():
            print(f"\n2. Testing parameter: {param_name}")
            param_results = {}
            
            for test_value in test_values:
                print(f"   Testing {param_name} = {test_value}")
                
                # Create parameter override
                param_override = {param_name: test_value}
                
                # Run optimization with this parameter value
                result = self._run_single_optimization(
                    f"{param_name}_{test_value}", 
                    param_override, 
                    max_iterations, 
                    time_limit
                )
                
                param_results[test_value] = result
                
                # Quick progress indicator
                cost_change = ((result['cost'] - baseline_result['cost']) / 
                              baseline_result['cost'] * 100)
                print(f"      Cost: €{result['cost']:,.0f} ({cost_change:+.1f}%)")
            
            sensitivity_results['parameter_tests'][param_name] = param_results
        
        self.sensitivity_results = sensitivity_results
        return sensitivity_results
    
    def run_robustness_analysis(self,
                               demand_scenarios: List[float] = None,
                               distance_perturbations: List[float] = None,
                               max_iterations: int = 2000,
                               time_limit: int = 600) -> Dict:
        """
        Test robustness of optimal networks under different scenarios.
        
        Args:
            demand_scenarios: List of demand multipliers (e.g., [0.8, 1.0, 1.2])
            distance_perturbations: List of distance noise levels (e.g., [0.0, 0.1, 0.2])
            max_iterations: Max iterations per test
            time_limit: Time limit per test
            
        Returns:
            Dictionary containing robustness analysis results
        """
        print("\n=== NETWORK ROBUSTNESS ANALYSIS ===")
        
        # Default scenarios if not provided
        if demand_scenarios is None:
            demand_scenarios = [0.7, 0.85, 1.0, 1.15, 1.3]  # ±30% demand variation
            
        if distance_perturbations is None:
            distance_perturbations = [0.0, 0.05, 0.1, 0.15, 0.2]  # Up to 20% distance noise
        
        robustness_results = {
            'demand_scenarios': {},
            'distance_perturbations': {},
            'combined_stress_test': {}
        }
        
        # 1. Test demand scenario robustness
        print("\n1. Testing demand scenario robustness...")
        for demand_multiplier in demand_scenarios:
            print(f"   Testing demand scenario: {demand_multiplier:.1f}x")
            
            result = self._run_demand_scenario_test(
                demand_multiplier, max_iterations, time_limit
            )
            robustness_results['demand_scenarios'][demand_multiplier] = result
            
            print(f"      Cost: €{result['cost']:,.0f}, "
                  f"SPs: {result['n_service_points']}, "
                  f"APLs: {result['n_apls']}")
        
        # 2. Test distance perturbation robustness
        print("\n2. Testing distance perturbation robustness...")
        for noise_level in distance_perturbations:
            print(f"   Testing distance noise: {noise_level:.1%}")
            
            result = self._run_distance_perturbation_test(
                noise_level, max_iterations, time_limit
            )
            robustness_results['distance_perturbations'][noise_level] = result
            
            print(f"      Cost: €{result['cost']:,.0f}, "
                  f"Avg distance change: {result.get('avg_distance_change', 0):.1%}")
        
        # 3. Combined stress test (high demand + distance uncertainty)
        print("\n3. Running combined stress tests...")
        stress_scenarios = [
            (1.2, 0.1),  # High demand + some distance uncertainty
            (0.8, 0.15), # Low demand + higher distance uncertainty  
            (1.3, 0.2)   # Very high demand + high distance uncertainty
        ]
        
        for demand_mult, distance_noise in stress_scenarios:
            scenario_name = f"demand_{demand_mult:.1f}_noise_{distance_noise:.1%}"
            print(f"   Testing combined scenario: {scenario_name}")
            
            result = self._run_combined_stress_test(
                demand_mult, distance_noise, max_iterations, time_limit
            )
            robustness_results['combined_stress_test'][scenario_name] = result
            
            print(f"      Cost: €{result['cost']:,.0f}")
        
        self.robustness_results = robustness_results
        return robustness_results
    
    def _run_single_optimization(self, 
                            test_name: str, 
                            param_overrides: Dict, 
                            max_iterations: int, 
                            time_limit: int) -> Dict:
        """Run a single optimization with parameter overrides."""
    
        # Create a copy of the optimizer
        test_optimizer = EnhancedServicePointOptimizer()
    
        # Copy core data structures
        test_optimizer.nodes_data = self.base_optimizer.nodes_data.copy() if self.base_optimizer.nodes_data is not None else None
        test_optimizer.demand_data = copy.deepcopy(self.base_optimizer.demand_data)
        test_optimizer.distance_matrix = self.base_optimizer.distance_matrix.copy()
        test_optimizer.demographic_data = self.base_optimizer.demographic_data.copy() if self.base_optimizer.demographic_data is not None else None
    
        # Copy parameters and apply overrides
        test_optimizer.parameters = copy.deepcopy(self.base_parameters)
        test_optimizer.apl_fixed_cost = self.base_apl_params['apl_fixed_cost']
        test_optimizer.apl_capacity = self.base_apl_params['apl_capacity']
        test_optimizer.pickup_ratio = self.base_demand_split['pickup_ratio']
        test_optimizer.delivery_ratio = self.base_demand_split['delivery_ratio']
    
        # Apply parameter overrides
        for param_name, param_value in param_overrides.items():
            if param_name in test_optimizer.parameters:
                test_optimizer.parameters[param_name] = param_value
            elif param_name == 'apl_fixed_cost':
                test_optimizer.apl_fixed_cost = param_value
            elif param_name == 'apl_capacity':
                test_optimizer.apl_capacity = param_value
            elif param_name == 'pickup_ratio':
                test_optimizer.pickup_ratio = param_value
                test_optimizer.delivery_ratio = 1.0 - param_value
                # Recalculate demand data with new split
                self._recalculate_demand_split(test_optimizer)
    
        # Copy other essential attributes
        test_optimizer.candidate_locations = self.base_optimizer.candidate_locations.copy()
        test_optimizer.customer_nodes = self.base_optimizer.customer_nodes.copy()
        test_optimizer.node_to_idx = self.base_optimizer.node_to_idx.copy()
        test_optimizer.idx_to_node = self.base_optimizer.idx_to_node.copy()
        test_optimizer.apl_locations = self.base_optimizer.apl_locations.copy()
    
        # Run optimization
        start_time = time.time()
        try:
            success = test_optimizer.optimize(
                initial_temp=1000,  # Increased from 200 to avoid local optima
                cooling_rate=0.995,
                max_iterations=max_iterations,
                time_limit=time_limit,
                initial_strategy='random'
            )
            
            if success and test_optimizer.best_solution is not None:
                # Extract solution metrics
                solution = test_optimizer.best_solution
                y = solution['y']
                apl_open = solution.get('apl_open', np.zeros(len(test_optimizer.apl_locations), dtype=bool))
                
                result = {
                    'success': True,
                    'cost': test_optimizer.best_cost,
                    'n_service_points': np.sum(y),
                    'n_apls': np.sum(apl_open),
                    'total_capacity': np.sum(solution['capacity']),
                    'optimization_time': time.time() - start_time,
                    'solution': solution,
                    'parameters_used': param_overrides
                }
                
                # Calculate additional metrics
                result.update(self._calculate_solution_metrics(test_optimizer, solution))
                
            else:
                result = {
                    'success': False,
                    'cost': float('inf'),
                    'error': 'Optimization failed',
                    'optimization_time': time.time() - start_time,
                    'parameters_used': param_overrides
                }
                
        except Exception as e:
            result = {
                'success': False,
                'cost': float('inf'),
                'error': str(e),
                'optimization_time': time.time() - start_time,
                'parameters_used': param_overrides
            }
    
        return result
    
    def _run_demand_scenario_test(self, demand_multiplier: float, max_iterations: int, time_limit: int) -> Dict:
        """Test with modified demand levels."""
        
        # Create demand override by scaling existing demand
        demand_override = {}
        for node_id, demand_dict in self.base_optimizer.demand_data.items():
            demand_override[node_id] = {
                'total': demand_dict['total'] * demand_multiplier,
                'pickup': demand_dict['pickup'] * demand_multiplier,
                'delivery': demand_dict['delivery'] * demand_multiplier
            }
        
        # Create temporary optimizer with modified demand
        test_optimizer = EnhancedServicePointOptimizer()
        
        # Copy all attributes
        test_optimizer.nodes_data = self.base_optimizer.nodes_data.copy() if self.base_optimizer.nodes_data is not None else None
        test_optimizer.demand_data = demand_override
        test_optimizer.distance_matrix = self.base_optimizer.distance_matrix.copy()
        test_optimizer.demographic_data = self.base_optimizer.demographic_data.copy() if self.base_optimizer.demographic_data is not None else None
        test_optimizer.parameters = copy.deepcopy(self.base_parameters)
        test_optimizer.apl_fixed_cost = self.base_apl_params['apl_fixed_cost']
        test_optimizer.apl_capacity = self.base_apl_params['apl_capacity']
        test_optimizer.pickup_ratio = self.base_demand_split['pickup_ratio']
        test_optimizer.delivery_ratio = self.base_demand_split['delivery_ratio']
        test_optimizer.candidate_locations = self.base_optimizer.candidate_locations.copy()
        test_optimizer.customer_nodes = self.base_optimizer.customer_nodes.copy()
        test_optimizer.node_to_idx = self.base_optimizer.node_to_idx.copy()
        test_optimizer.idx_to_node = self.base_optimizer.idx_to_node.copy()
        test_optimizer.apl_locations = self.base_optimizer.apl_locations.copy()
        
        # Run optimization
        start_time = time.time()
        try:
            success = test_optimizer.optimize(
                initial_temp=5000,
                cooling_rate=0.995,
                max_iterations=max_iterations,
                time_limit=time_limit,
                initial_strategy='random'
            )
            
            if success and test_optimizer.best_solution is not None:
                solution = test_optimizer.best_solution
                y = solution['y']
                apl_open = solution.get('apl_open', np.zeros(len(test_optimizer.apl_locations), dtype=bool))
                
                result = {
                    'success': True,
                    'cost': test_optimizer.best_cost,
                    'n_service_points': np.sum(y),
                    'n_apls': np.sum(apl_open),
                    'total_capacity': np.sum(solution['capacity']),
                    'optimization_time': time.time() - start_time,
                    'demand_multiplier': demand_multiplier,
                    'solution': solution
                }
                
                # Calculate additional metrics
                result.update(self._calculate_solution_metrics(test_optimizer, solution))
                
            else:
                result = {
                    'success': False,
                    'cost': float('inf'),
                    'error': 'Optimization failed',
                    'demand_multiplier': demand_multiplier
                }
                
        except Exception as e:
            result = {
                'success': False,
                'cost': float('inf'),
                'error': str(e),
                'demand_multiplier': demand_multiplier
            }
        
        return result
    
    def _run_distance_perturbation_test(self, noise_level: float, max_iterations: int, time_limit: int) -> Dict:
        """Test with perturbed distance matrix."""
        
        # Create perturbed distance matrix
        original_matrix = self.base_optimizer.distance_matrix.copy()
        
        # Add random noise (multiplicative)
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(1.0, noise_level, original_matrix.shape)
        
        # Apply noise only to finite distances
        perturbed_matrix = original_matrix.copy()
        finite_mask = np.isfinite(original_matrix)
        perturbed_matrix[finite_mask] *= noise[finite_mask]
        
        # Ensure non-negative distances and maintain diagonal zeros
        perturbed_matrix = np.maximum(perturbed_matrix, 0)
        np.fill_diagonal(perturbed_matrix, 0.0)
        
        # Calculate average distance change
        avg_distance_change = np.mean(np.abs(perturbed_matrix[finite_mask] - original_matrix[finite_mask]) / 
                                    original_matrix[finite_mask])
        
        # Create temporary optimizer with perturbed distances
        test_optimizer = EnhancedServicePointOptimizer()
        
        # Copy all attributes
        test_optimizer.nodes_data = self.base_optimizer.nodes_data.copy() if self.base_optimizer.nodes_data is not None else None
        test_optimizer.demand_data = copy.deepcopy(self.base_optimizer.demand_data)
        test_optimizer.distance_matrix = perturbed_matrix
        test_optimizer.demographic_data = self.base_optimizer.demographic_data.copy() if self.base_optimizer.demographic_data is not None else None
        test_optimizer.parameters = copy.deepcopy(self.base_parameters)
        test_optimizer.apl_fixed_cost = self.base_apl_params['apl_fixed_cost']
        test_optimizer.apl_capacity = self.base_apl_params['apl_capacity']
        test_optimizer.pickup_ratio = self.base_demand_split['pickup_ratio']
        test_optimizer.delivery_ratio = self.base_demand_split['delivery_ratio']
        test_optimizer.candidate_locations = self.base_optimizer.candidate_locations.copy()
        test_optimizer.customer_nodes = self.base_optimizer.customer_nodes.copy()
        test_optimizer.node_to_idx = self.base_optimizer.node_to_idx.copy()
        test_optimizer.idx_to_node = self.base_optimizer.idx_to_node.copy()
        test_optimizer.apl_locations = self.base_optimizer.apl_locations.copy()
        
        # Run optimization
        start_time = time.time()
        try:
            success = test_optimizer.optimize(
                initial_temp=5000,
                cooling_rate=0.995,
                max_iterations=max_iterations,
                time_limit=time_limit,
                initial_strategy='random'
            )
            
            if success and test_optimizer.best_solution is not None:
                solution = test_optimizer.best_solution
                y = solution['y']
                apl_open = solution.get('apl_open', np.zeros(len(test_optimizer.apl_locations), dtype=bool))
                
                result = {
                    'success': True,
                    'cost': test_optimizer.best_cost,
                    'n_service_points': np.sum(y),
                    'n_apls': np.sum(apl_open),
                    'total_capacity': np.sum(solution['capacity']),
                    'optimization_time': time.time() - start_time,
                    'noise_level': noise_level,
                    'avg_distance_change': avg_distance_change,
                    'solution': solution
                }
                
                # Calculate additional metrics
                result.update(self._calculate_solution_metrics(test_optimizer, solution))
                
            else:
                result = {
                    'success': False,
                    'cost': float('inf'),
                    'error': 'Optimization failed',
                    'noise_level': noise_level,
                    'avg_distance_change': avg_distance_change
                }
                
        except Exception as e:
            result = {
                'success': False,
                'cost': float('inf'),
                'error': str(e),
                'noise_level': noise_level,
                'avg_distance_change': avg_distance_change
            }
        
        return result
    
    def _run_combined_stress_test(self, demand_multiplier: float, noise_level: float, 
                                 max_iterations: int, time_limit: int) -> Dict:
        """Test with both demand changes and distance perturbations."""
        
        # Create modified demand
        demand_override = {}
        for node_id, demand_dict in self.base_optimizer.demand_data.items():
            demand_override[node_id] = {
                'total': demand_dict['total'] * demand_multiplier,
                'pickup': demand_dict['pickup'] * demand_multiplier,
                'delivery': demand_dict['delivery'] * demand_multiplier
            }
        
        # Create perturbed distance matrix
        original_matrix = self.base_optimizer.distance_matrix.copy()
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(1.0, noise_level, original_matrix.shape)
        
        perturbed_matrix = original_matrix.copy()
        finite_mask = np.isfinite(original_matrix)
        perturbed_matrix[finite_mask] *= noise[finite_mask]
        perturbed_matrix = np.maximum(perturbed_matrix, 0)
        np.fill_diagonal(perturbed_matrix, 0.0)
        
        # Create temporary optimizer
        test_optimizer = EnhancedServicePointOptimizer()
        
        # Copy all attributes with modifications
        test_optimizer.nodes_data = self.base_optimizer.nodes_data.copy() if self.base_optimizer.nodes_data is not None else None
        test_optimizer.demand_data = demand_override
        test_optimizer.distance_matrix = perturbed_matrix
        test_optimizer.demographic_data = self.base_optimizer.demographic_data.copy() if self.base_optimizer.demographic_data is not None else None
        test_optimizer.parameters = copy.deepcopy(self.base_parameters)
        test_optimizer.apl_fixed_cost = self.base_apl_params['apl_fixed_cost']
        test_optimizer.apl_capacity = self.base_apl_params['apl_capacity']
        test_optimizer.pickup_ratio = self.base_demand_split['pickup_ratio']
        test_optimizer.delivery_ratio = self.base_demand_split['delivery_ratio']
        test_optimizer.candidate_locations = self.base_optimizer.candidate_locations.copy()
        test_optimizer.customer_nodes = self.base_optimizer.customer_nodes.copy()
        test_optimizer.node_to_idx = self.base_optimizer.node_to_idx.copy()
        test_optimizer.idx_to_node = self.base_optimizer.idx_to_node.copy()
        test_optimizer.apl_locations = self.base_optimizer.apl_locations.copy()
        
        # Run optimization
        start_time = time.time()
        try:
            success = test_optimizer.optimize(
                initial_temp=5000,
                cooling_rate=0.995,
                max_iterations=max_iterations,
                time_limit=time_limit,
                initial_strategy='random'
            )
            
            if success and test_optimizer.best_solution is not None:
                solution = test_optimizer.best_solution
                y = solution['y']
                apl_open = solution.get('apl_open', np.zeros(len(test_optimizer.apl_locations), dtype=bool))
                
                result = {
                    'success': True,
                    'cost': test_optimizer.best_cost,
                    'n_service_points': np.sum(y),
                    'n_apls': np.sum(apl_open),
                    'total_capacity': np.sum(solution['capacity']),
                    'optimization_time': time.time() - start_time,
                    'demand_multiplier': demand_multiplier,
                    'noise_level': noise_level,
                    'solution': solution
                }
                
                # Calculate additional metrics
                result.update(self._calculate_solution_metrics(test_optimizer, solution))
                
            else:
                result = {
                    'success': False,
                    'cost': float('inf'),
                    'error': 'Optimization failed',
                    'demand_multiplier': demand_multiplier,
                    'noise_level': noise_level
                }
                
        except Exception as e:
            result = {
                'success': False,
                'cost': float('inf'),
                'error': str(e),
                'demand_multiplier': demand_multiplier,
                'noise_level': noise_level
            }
        
        return result
    
    def _calculate_solution_metrics(self, optimizer: EnhancedServicePointOptimizer, solution: Dict) -> Dict:
        """Calculate additional solution metrics for analysis."""
        
        y = solution['y']
        x_assigned = solution['x_assigned']
        capacity = solution['capacity']
        apl_open = solution.get('apl_open', np.zeros(len(optimizer.apl_locations), dtype=bool))
        
        metrics = {}
        
        # Calculate average distance to assigned facilities
        total_distance = 0
        total_customers = 0
        
        for cust_idx in range(len(optimizer.customer_nodes)):
            cust_id = optimizer.customer_nodes[cust_idx]
            cust_node_idx = optimizer.node_to_idx[cust_id]
            
            # Check SP assignments
            for loc_idx in range(len(optimizer.candidate_locations)):
                if x_assigned[cust_idx, loc_idx]:
                    loc_id = optimizer.candidate_locations[loc_idx]
                    loc_node_idx = optimizer.node_to_idx[loc_id]
                    distance = optimizer.distance_matrix[cust_node_idx, loc_node_idx]
                    total_distance += distance
                    total_customers += 1
                    break
            
            # Check APL assignments
            for apl_idx in range(len(optimizer.apl_locations)):
                assign_idx = len(optimizer.candidate_locations) + apl_idx
                if x_assigned[cust_idx, assign_idx]:
                    apl_id = optimizer.apl_locations[apl_idx]
                    apl_node_idx = optimizer.node_to_idx[apl_id]
                    distance = optimizer.distance_matrix[cust_node_idx, apl_node_idx]
                    total_distance += distance
                    total_customers += 1
                    break
        
        metrics['avg_distance_to_facility'] = total_distance / max(total_customers, 1)
        
        # Calculate capacity utilization
        total_capacity = np.sum(capacity)
        total_demand = sum(data['total'] for data in optimizer.demand_data.values())
        metrics['capacity_utilization'] = total_demand / max(total_capacity, 1)
        
        # Calculate bounce rates for SPs
        total_pickup_demand = 0
        total_bounced = 0
        
        for loc_idx in range(len(optimizer.candidate_locations)):
            if y[loc_idx]:
                pickup_demand = 0
                for cust_idx in range(len(optimizer.customer_nodes)):
                    if x_assigned[cust_idx, loc_idx]:
                        cust_id = optimizer.customer_nodes[cust_idx]
                        pickup_demand += optimizer.demand_data[cust_id]['pickup']
                
                if pickup_demand > 0:
                    total_pickup_demand += pickup_demand
                    bounce = max(0, pickup_demand - capacity[loc_idx])
                    total_bounced += bounce
        
        metrics['city_bounce_rate'] = total_bounced / max(total_pickup_demand, 1)
        
        return metrics
    
    def _recalculate_demand_split(self, optimizer: EnhancedServicePointOptimizer):
        """Recalculate demand split when pickup_ratio changes."""
        for node_id, demand_dict in optimizer.demand_data.items():
            total_demand = demand_dict['total']
            optimizer.demand_data[node_id] = {
                'total': total_demand,
                'pickup': total_demand * optimizer.pickup_ratio,
                'delivery': total_demand * optimizer.delivery_ratio
            }
    
    def generate_sensitivity_report(self, save_path: str = None) -> str:
        """Generate comprehensive sensitivity analysis report."""
        
        if not self.sensitivity_results:
            return "No sensitivity analysis results available. Run run_parameter_sensitivity_analysis() first."
        
        report = []
        report.append("=" * 80)
        report.append("POST&L SENSITIVITY ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Baseline results
        baseline = self.sensitivity_results['baseline']
        report.append(f"\nBASELINE SOLUTION:")
        report.append(f"  Total Cost: €{baseline['cost']:,.0f}")
        report.append(f"  Service Points: {baseline.get('n_service_points', 'N/A')}")
        report.append(f"  APLs: {baseline.get('n_apls', 'N/A')}")
        report.append(f"  Average Distance: {baseline.get('avg_distance_to_facility', 0):.2f} km")
        report.append(f"  City Bounce Rate: {baseline.get('city_bounce_rate', 0):.1%}")
        
        # Parameter sensitivity results
        report.append(f"\nPARAMETER SENSITIVITY RESULTS:")
        report.append("-" * 50)
        
        for param_name, param_results in self.sensitivity_results['parameter_tests'].items():
            report.append(f"\nParameter: {param_name}")
            
            # Find min and max costs
            costs = [result['cost'] for result in param_results.values() if result['success']]
            if costs:
                min_cost = min(costs)
                max_cost = max(costs)
                cost_range = ((max_cost - min_cost) / baseline['cost']) * 100
                
                report.append(f"  Cost Range: €{min_cost:,.0f} - €{max_cost:,.0f} ({cost_range:.1f}% variation)")
                
                # Find most sensitive values
                max_impact = 0
                best_value = None
                worst_value = None
                
                for test_value, result in param_results.items():
                    if result['success']:
                        impact = abs((result['cost'] - baseline['cost']) / baseline['cost'] * 100)
                        if impact > max_impact:
                            max_impact = impact
                        
                        if result['cost'] == min_cost:
                            best_value = test_value
                        if result['cost'] == max_cost:
                            worst_value = test_value
                
                report.append(f"  Maximum Impact: {max_impact:.1f}%")
                report.append(f"  Best Value: {best_value} (€{min_cost:,.0f})")
                report.append(f"  Worst Value: {worst_value} (€{max_cost:,.0f})")
            else:
                report.append(f"  No successful optimizations for this parameter")
        
        # Robustness results
        if self.robustness_results:
            report.append(f"\nROBUSTNESS ANALYSIS RESULTS:")
            report.append("-" * 50)
            
            # Demand scenario robustness
            if 'demand_scenarios' in self.robustness_results:
                report.append(f"\nDemand Scenario Robustness:")
                demand_costs = []
                for demand_mult, result in self.robustness_results['demand_scenarios'].items():
                    if result['success']:
                        cost_change = ((result['cost'] - baseline['cost']) / baseline['cost']) * 100
                        report.append(f"  {demand_mult:.1f}x demand: €{result['cost']:,.0f} ({cost_change:+.1f}%)")
                        demand_costs.append(result['cost'])
                
                if demand_costs:
                    demand_volatility = (max(demand_costs) - min(demand_costs)) / baseline['cost'] * 100
                    report.append(f"  Demand Volatility: {demand_volatility:.1f}%")
            
            # Distance perturbation robustness
            if 'distance_perturbations' in self.robustness_results:
                report.append(f"\nDistance Perturbation Robustness:")
                for noise_level, result in self.robustness_results['distance_perturbations'].items():
                    if result['success']:
                        cost_change = ((result['cost'] - baseline['cost']) / baseline['cost']) * 100
                        report.append(f"  {noise_level:.1%} noise: €{result['cost']:,.0f} ({cost_change:+.1f}%)")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Sensitivity report saved to {save_path}")
        
        return report_text
    
    def plot_sensitivity_analysis(self, save_path: str = None):
        """Create comprehensive sensitivity analysis visualizations."""
        
        if not self.sensitivity_results:
            print("No sensitivity results to plot. Run sensitivity analysis first.")
            return
        
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Plot 1: Parameter sensitivity tornado chart
            ax1 = plt.subplot(2, 3, 1)
            try:
                self._plot_tornado_chart(ax1)
            except Exception as e:
                ax1.text(0.5, 0.5, f'Error in tornado chart:\n{str(e)}', 
                        ha='center', va='center', transform=ax1.transAxes)
                print(f"Warning: Error in tornado chart: {e}")
            
            # Plot 2: Cost vs parameter value plots
            ax2 = plt.subplot(2, 3, 2)
            try:
                self._plot_cost_vs_parameters(ax2)
            except Exception as e:
                ax2.text(0.5, 0.5, f'Error in cost vs parameters:\n{str(e)}', 
                        ha='center', va='center', transform=ax2.transAxes)
                print(f"Warning: Error in cost vs parameters plot: {e}")
            
            # Plot 3: Network size sensitivity
            ax3 = plt.subplot(2, 3, 3)
            try:
                self._plot_network_size_sensitivity(ax3)
            except Exception as e:
                ax3.text(0.5, 0.5, f'Error in network size plot:\n{str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
                print(f"Warning: Error in network size sensitivity plot: {e}")
            
            # Plot 4-6: Robustness analysis (if available)
            if self.robustness_results:
                ax4 = plt.subplot(2, 3, 4)
                try:
                    self._plot_demand_robustness(ax4)
                except Exception as e:
                    ax4.text(0.5, 0.5, f'Error in demand robustness:\n{str(e)}', 
                            ha='center', va='center', transform=ax4.transAxes)
                    print(f"Warning: Error in demand robustness plot: {e}")
                
                ax5 = plt.subplot(2, 3, 5)
                try:
                    self._plot_distance_robustness(ax5)
                except Exception as e:
                    ax5.text(0.5, 0.5, f'Error in distance robustness:\n{str(e)}', 
                            ha='center', va='center', transform=ax5.transAxes)
                    print(f"Warning: Error in distance robustness plot: {e}")
                
                ax6 = plt.subplot(2, 3, 6)
                try:
                    self._plot_robustness_heatmap(ax6)
                except Exception as e:
                    ax6.text(0.5, 0.5, f'Error in robustness heatmap:\n{str(e)}', 
                            ha='center', va='center', transform=ax6.transAxes)
                    print(f"Warning: Error in robustness heatmap: {e}")
            else:
                # Fill empty plots with info text
                for i, ax in enumerate([plt.subplot(2, 3, 4), plt.subplot(2, 3, 5), plt.subplot(2, 3, 6)]):
                    ax.text(0.5, 0.5, 'No robustness analysis data\nRun robustness analysis first', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(['Demand Robustness', 'Distance Robustness', 'Stress Test Heatmap'][i])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Sensitivity plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating sensitivity analysis plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_tornado_chart(self, ax):
        """Create tornado chart showing parameter sensitivity."""
        baseline_cost = self.sensitivity_results['baseline']['cost']
        
        param_impacts = {}
        
        for param_name, param_results in self.sensitivity_results['parameter_tests'].items():
            costs = [result['cost'] for result in param_results.values() if result['success']]
            if costs:
                min_cost = min(costs)
                max_cost = max(costs)
                
                # Calculate percentage impact from baseline
                low_impact = (min_cost - baseline_cost) / baseline_cost * 100
                high_impact = (max_cost - baseline_cost) / baseline_cost * 100
                
                param_impacts[param_name] = (low_impact, high_impact)
        
        # Sort by total impact range
        sorted_params = sorted(param_impacts.items(), 
                              key=lambda x: abs(x[1][1] - x[1][0]), 
                              reverse=True)
        
        y_pos = np.arange(len(sorted_params))
        
        for i, (param_name, (low, high)) in enumerate(sorted_params):
            # Plot bars
            ax.barh(i, low, height=0.6, color='red', alpha=0.7, label='Low' if i == 0 else '')
            ax.barh(i, high, height=0.6, color='green', alpha=0.7, label='High' if i == 0 else '')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([param[0] for param in sorted_params])
        ax.set_xlabel('Cost Impact (%)')
        ax.set_title('Parameter Sensitivity (Tornado Chart)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cost_vs_parameters(self, ax):
        """Plot cost vs parameter values."""
        baseline_cost = self.sensitivity_results['baseline']['cost']
        
        # Select most impactful parameters
        param_impacts = {}
        for param_name, param_results in self.sensitivity_results['parameter_tests'].items():
            costs = [result['cost'] for result in param_results.values() if result['success']]
            if costs:
                impact_range = (max(costs) - min(costs)) / baseline_cost * 100
                param_impacts[param_name] = impact_range
        
        # Get top 3 most impactful parameters
        top_params = sorted(param_impacts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        colors = ['blue', 'red', 'green']
        
        for i, (param_name, _) in enumerate(top_params):
            param_results = self.sensitivity_results['parameter_tests'][param_name]
            
            test_values = []
            costs = []
            
            for test_value, result in param_results.items():
                if result['success']:
                    test_values.append(test_value)
                    costs.append(result['cost'])
            
            if test_values:
                # Sort by test value
                sorted_data = sorted(zip(test_values, costs))
                test_values, costs = zip(*sorted_data)
                
                ax.plot(test_values, costs, 'o-', color=colors[i], 
                       label=param_name, linewidth=2, markersize=6)
        
        ax.axhline(y=baseline_cost, color='black', linestyle='--', alpha=0.7, label='Baseline')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Total Cost (€)')
        ax.set_title('Cost vs Parameter Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_network_size_sensitivity(self, ax):
        """Plot network size vs parameters."""
        baseline_sps = self.sensitivity_results['baseline'].get('n_service_points', 0)
        baseline_apls = self.sensitivity_results['baseline'].get('n_apls', 0)
        
        param_names = []
        sp_ranges = []
        apl_ranges = []
        
        for param_name, param_results in self.sensitivity_results['parameter_tests'].items():
            sps = [result.get('n_service_points', 0) for result in param_results.values() if result['success']]
            apls = [result.get('n_apls', 0) for result in param_results.values() if result['success']]
            
            if sps and apls:
                param_names.append(param_name)
                sp_ranges.append((min(sps), max(sps)))
                apl_ranges.append((min(apls), max(apls)))
    
        x = np.arange(len(param_names))
        width = 0.35
        
        # Plot SP ranges - Fix error bar calculation
        sp_lows = [r[0] for r in sp_ranges]
        sp_highs = [r[1] for r in sp_ranges]
        
        # Calculate error bars correctly (always positive)
        sp_errors_lower = [max(0, baseline_sps - low) for low in sp_lows]  # Distance below baseline
        sp_errors_upper = [max(0, high - baseline_sps) for high in sp_highs]  # Distance above baseline
        sp_errors = [sp_errors_lower, sp_errors_upper]
        
        ax.bar(x - width/2, [baseline_sps] * len(param_names), width, 
               yerr=sp_errors, label='Service Points', alpha=0.7, capsize=5)
    
        # Plot APL ranges - Fix error bar calculation
        apl_lows = [r[0] for r in apl_ranges]
        apl_highs = [r[1] for r in apl_ranges]
        
        # Calculate error bars correctly (always positive)
        apl_errors_lower = [max(0, baseline_apls - low) for low in apl_lows]  # Distance below baseline
        apl_errors_upper = [max(0, high - baseline_apls) for high in apl_highs]  # Distance above baseline
        apl_errors = [apl_errors_lower, apl_errors_upper]
        
        ax.bar(x + width/2, [baseline_apls] * len(param_names), width,
               yerr=apl_errors, label='APLs', alpha=0.7, capsize=5)
        
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Number of Facilities')
        ax.set_title('Network Size Sensitivity')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_demand_robustness(self, ax):
        """Plot demand scenario robustness."""
        if 'demand_scenarios' not in self.robustness_results:
            ax.text(0.5, 0.5, 'No demand robustness data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        demand_mults = []
        costs = []
        n_sps = []
        n_apls = []
        
        for demand_mult, result in self.robustness_results['demand_scenarios'].items():
            if result['success']:
                demand_mults.append(demand_mult)
                costs.append(result['cost'])
                n_sps.append(result.get('n_service_points', 0))
                n_apls.append(result.get('n_apls', 0))
        
        if demand_mults:
            # Sort by demand multiplier
            sorted_data = sorted(zip(demand_mults, costs, n_sps, n_apls))
            demand_mults, costs, n_sps, n_apls = zip(*sorted_data)
            
            ax2 = ax.twinx()
            
            # Plot cost
            line1 = ax.plot(demand_mults, costs, 'b-o', linewidth=2, markersize=6, label='Total Cost')
            ax.set_xlabel('Demand Multiplier')
            ax.set_ylabel('Total Cost (€)', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            # Plot network size
            line2 = ax2.plot(demand_mults, n_sps, 'r-s', linewidth=2, markersize=6, label='Service Points')
            line3 = ax2.plot(demand_mults, n_apls, 'g-^', linewidth=2, markersize=6, label='APLs')
            ax2.set_ylabel('Number of Facilities', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combined legend
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.set_title('Demand Scenario Robustness')
            ax.grid(True, alpha=0.3)
    
    def _plot_distance_robustness(self, ax):
        """Plot distance perturbation robustness."""
        if 'distance_perturbations' not in self.robustness_results:
            ax.text(0.5, 0.5, 'No distance robustness data available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        noise_levels = []
        costs = []
        avg_distances = []
        
        for noise_level, result in self.robustness_results['distance_perturbations'].items():
            if result['success']:
                noise_levels.append(noise_level * 100)  # Convert to percentage
                costs.append(result['cost'])
                avg_distances.append(result.get('avg_distance_to_facility', 0))
        
        if noise_levels:
            # Sort by noise level
            sorted_data = sorted(zip(noise_levels, costs, avg_distances))
            noise_levels, costs, avg_distances = zip(*sorted_data)
            
            ax2 = ax.twinx()
            
            # Plot cost
            line1 = ax.plot(noise_levels, costs, 'b-o', linewidth=2, markersize=6, label='Total Cost')
            ax.set_xlabel('Distance Noise Level (%)')
            ax.set_ylabel('Total Cost (€)', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            # Plot average distance
            line2 = ax2.plot(noise_levels, avg_distances, 'r-s', linewidth=2, markersize=6, label='Avg Distance')
            ax2.set_ylabel('Average Distance (km)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.set_title('Distance Perturbation Robustness')
            ax.grid(True, alpha=0.3)
    
    def _plot_robustness_heatmap(self, ax):
        """Plot combined stress test results as heatmap."""
        if 'combined_stress_test' not in self.robustness_results:
            ax.text(0.5, 0.5, 'No stress test data available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract data for heatmap
        demand_mults = set()
        noise_levels = set()
        cost_matrix = {}
        
        for scenario_name, result in self.robustness_results['combined_stress_test'].items():
            if result['success']:
                parts = scenario_name.split('_')
                demand_mult = float(parts[1])
                noise_level = float(parts[3].replace('%', '')) / 100
                
                demand_mults.add(demand_mult)
                noise_levels.add(noise_level)
                cost_matrix[(demand_mult, noise_level)] = result['cost']
        
        if cost_matrix:
            demand_mults = sorted(demand_mults)
            noise_levels = sorted(noise_levels)
            
            # Create matrix
            matrix = np.zeros((len(noise_levels), len(demand_mults)))
            for i, noise in enumerate(noise_levels):
                for j, demand in enumerate(demand_mults):
                    if (demand, noise) in cost_matrix:
                        matrix[i, j] = cost_matrix[(demand, noise)]
                    else:
                        matrix[i, j] = np.nan
            
            # Plot heatmap
            im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(demand_mults)))
            ax.set_xticklabels([f'{d:.1f}' for d in demand_mults])
            ax.set_yticks(range(len(noise_levels)))
            ax.set_yticklabels([f'{n:.0%}' for n in noise_levels])
            
            ax.set_xlabel('Demand Multiplier')
            ax.set_ylabel('Distance Noise Level')
            ax.set_title('Stress Test Results\n(Total Cost)')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Total Cost (€)')
            
            # Add text annotations
            for i in range(len(noise_levels)):
                for j in range(len(demand_mults)):
                    if not np.isnan(matrix[i, j]):
                        text = ax.text(j, i, f'{matrix[i, j]/1000:.0f}K',
                                     ha="center", va="center", color="white", fontweight='bold')

    def export_results_to_excel(self, file_path: str):
        """Export all sensitivity and robustness results to Excel."""
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            
            # Export baseline results
            if self.sensitivity_results:
                baseline_df = pd.DataFrame([self.sensitivity_results['baseline']])
                baseline_df.to_excel(writer, sheet_name='Baseline', index=False)
                
                # Export parameter sensitivity results
                for param_name, param_results in self.sensitivity_results['parameter_tests'].items():
                    param_df = pd.DataFrame.from_dict(param_results, orient='index')
                    param_df.index.name = 'Parameter_Value'
                    param_df.to_excel(writer, sheet_name=f'Sensitivity_{param_name}')
            
            # Export robustness results
            if self.robustness_results:
                
                # Demand scenarios
                if 'demand_scenarios' in self.robustness_results:
                    demand_df = pd.DataFrame.from_dict(
                        self.robustness_results['demand_scenarios'], orient='index'
                    )
                    demand_df.index.name = 'Demand_Multiplier'
                    demand_df.to_excel(writer, sheet_name='Demand_Robustness')
                
                # Distance perturbations
                if 'distance_perturbations' in self.robustness_results:
                    distance_df = pd.DataFrame.from_dict(
                        self.robustness_results['distance_perturbations'], orient='index'
                    )
                    distance_df.index.name = 'Noise_Level'
                    distance_df.to_excel(writer, sheet_name='Distance_Robustness')
                
                # Stress tests
                if 'combined_stress_test' in self.robustness_results:
                    stress_df = pd.DataFrame.from_dict(
                        self.robustness_results['combined_stress_test'], orient='index'
                    )
                    stress_df.index.name = 'Scenario'
                    stress_df.to_excel(writer, sheet_name='Stress_Tests')
        
        print(f"Results exported to {file_path}")


# Example usage and integration with main optimization
if __name__ == "__main__":
    print("=== POST&L SENSITIVITY ANALYSIS ===")
    
    try:
        # First run the base optimization
        print("1. Setting up base optimizer...")
        base_optimizer = EnhancedServicePointOptimizer()
        
        # Load data
        base_optimizer.load_data(
            'data_Maastricht_2024.xlsx',
            'distance_matrix.csv',
            'node_to_service_point_assignments_max5kmh.csv'
        )
        
        # Run a quick baseline optimization
        print("2. Running baseline optimization...")
        success = base_optimizer.optimize(
            initial_temp=5000,
            cooling_rate=0.995,
            max_iterations=10000,
            time_limit=600,  # 10 minutes
            initial_strategy='existing'
        )
        
        if not success:
            print("Baseline optimization failed!")
            exit(1)
        
        print(f"Baseline optimization completed: €{base_optimizer.best_cost:,.0f}")
        
        # Initialize sensitivity analyzer
        print("3. Initializing sensitivity analyzer...")
        analyzer = SensitivityAnalyzer(base_optimizer)
        
        # Run parameter sensitivity analysis
        print("4. Running parameter sensitivity analysis...")
        sensitivity_results = analyzer.run_parameter_sensitivity_analysis(
            max_iterations=10000,
            time_limit=600  # 5 minutes per test
        )
        
        # Run robustness analysis
        print("5. Running robustness analysis...")
        robustness_results = analyzer.run_robustness_analysis(
            max_iterations=10000,
            time_limit=600  # 3 minutes per test
        )
        
        # Generate report
        print("6. Generatin6 analysis report...")
        report = analyzer.generate_sensitivity_report('sensitivity_report.txt')
        print("\nSENSITIVITY ANALYSIS SUMMARY:")
        print(report)
        
        # Create visualizations
        print("7. Creating visualizations...")
        analyzer.plot_sensitivity_analysis('sensitivity_analysis.png')
        
        # Export to Excel
        print("8. Exporting results to Excel...")
        analyzer.export_results_to_excel('sensitivity_analysis_results.xlsx')
        
        print("\n=== SENSITIVITY ANALYSIS COMPLETED ===")
        print("Files created:")
        print("- sensitivity_report.txt")
        print("- sensitivity_analysis.png")
        print("- sensitivity_analysis_results.xlsx")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
    except Exception as e:
        print(f"Error during sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()