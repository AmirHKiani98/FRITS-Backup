#!/usr/bin/env python3
"""
Example usage of the modular fixed timing simulation.
"""

from fixed import (
    SimulationConfig, 
    FixedTimingSimulation, 
    SUMOController, 
    TrafficDataCollector,
    DataLogger,
    create_default_config
)


def example_basic_simulation():
    """Example: Basic simulation with default parameters."""
    print("=== Basic Simulation Example ===")
    
    config = create_default_config(
        simulation_time=1000,  # Shorter simulation for example
        output_file="basic_simulation_results.csv"
    )
    
    simulation = FixedTimingSimulation(config)
    results = simulation.run_simulation()
    
    print(f"Simulation completed. Results saved to {config.output_file}")
    print(f"Number of data points collected: {len(results)}")


def example_custom_config():
    """Example: Custom configuration simulation."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = SimulationConfig(
        simulation_time=500,
        net_file="net/4x4.net.xml",
        route_file="rou/4x4c2c1.rou.xml",
        output_file="custom_simulation_results.csv",
        use_gui=False,  # Set to True to see visualization
        step_length=0.5  # Smaller time steps for more granular data
    )
    
    simulation = FixedTimingSimulation(config)
    results = simulation.run_simulation()
    
    # Analyze results
    peak_vehicles = results['system_total_vehicles'].max()
    avg_speed = results['system_mean_speed'].mean()
    
    print(f"Peak vehicles in simulation: {peak_vehicles}")
    print(f"Average speed throughout simulation: {avg_speed:.2f} m/s")


def example_custom_data_collection():
    """Example: Using individual components for custom data collection."""
    print("\n=== Custom Data Collection Example ===")
    
    config = create_default_config(
        simulation_time=300,
        output_file="custom_data_collection.csv"
    )
    
    controller = SUMOController(config)
    data_collector = TrafficDataCollector()
    logger = DataLogger(config.output_file)
    
    with controller:
        print("Collecting data every 10 steps...")
        
        for step in range(config.simulation_time):
            controller.step_simulation()
            
            # Collect data only every 10 steps
            if step % 10 == 0:
                metrics = data_collector.get_system_metrics()
                
                # Add custom metrics
                metrics['simulation_step'] = step
                metrics['vehicles_per_second'] = metrics['system_total_vehicles'] / (step + 1)
                
                logger.add_record(metrics)
                
                if step % 100 == 0:
                    print(f"Step {step}: {metrics['system_total_vehicles']} vehicles")
    
    logger.save_to_csv()
    print(f"Custom data collection completed. Saved to {config.output_file}")


def example_compare_scenarios():
    """Example: Compare multiple simulation scenarios."""
    print("\n=== Scenario Comparison Example ===")
    
    scenarios = [
        ("short_simulation", 500),
        ("medium_simulation", 1000),
        ("long_simulation", 1500)
    ]
    
    results_comparison = {}
    
    for scenario_name, sim_time in scenarios:
        print(f"Running {scenario_name}...")
        
        config = create_default_config(
            simulation_time=sim_time,
            output_file=f"{scenario_name}_results.csv"
        )
        
        simulation = FixedTimingSimulation(config)
        results = simulation.run_simulation()
        
        # Store summary statistics
        results_comparison[scenario_name] = {
            "simulation_time": sim_time,
            "avg_vehicles": results['system_total_vehicles'].mean(),
            "max_vehicles": results['system_total_vehicles'].max(),
            "avg_waiting_time": results['system_mean_waiting_time'].mean(),
            "avg_speed": results['system_mean_speed'].mean()
        }
    
    # Print comparison
    print("\n=== Scenario Comparison Results ===")
    for scenario, stats in results_comparison.items():
        print(f"\n{scenario}:")
        print(f"  Simulation time: {stats['simulation_time']} steps")
        print(f"  Average vehicles: {stats['avg_vehicles']:.2f}")
        print(f"  Peak vehicles: {stats['max_vehicles']:.0f}")
        print(f"  Average waiting time: {stats['avg_waiting_time']:.2f} seconds")
        print(f"  Average speed: {stats['avg_speed']:.2f} m/s")


def main():
    """Run all examples."""
    try:
        example_basic_simulation()
        example_custom_config()
        example_custom_data_collection()
        example_compare_scenarios()
        
        print("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure SUMO is properly installed and network files exist.")


if __name__ == "__main__":
    main()
