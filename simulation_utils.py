"""
Utility functions and extensions for the modular SUMO simulation framework.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SimulationAnalyzer:
    """Analyzes simulation results and generates insights."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
    
    def plot_metrics_over_time(self, 
                              metrics: Optional[List[str]] = None, 
                              save_path: Optional[str] = None) -> None:
        """Plot specified metrics over simulation time."""
        if metrics is None:
            metrics = ['system_total_vehicles', 'system_total_stopped', 'system_mean_speed']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in self.df.columns:
                axes[i].plot(self.df['system_time'], self.df[metric])
                axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
                axes[i].set_xlabel('Simulation Time (s)')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].grid(True)
            else:
                axes[i].text(0.5, 0.5, f'Metric "{metric}" not found', 
                           ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all numeric columns."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_cols:
            summary[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median()
            }
        
        return summary
    
    def detect_congestion_periods(self, 
                                stopped_threshold: int = 50,
                                duration_threshold: int = 30) -> List[Tuple[float, float]]:
        """Detect periods of high congestion."""
        high_congestion = self.df['system_total_stopped'] > stopped_threshold
        congestion_periods = []
        
        start_time = None
        for i, (time, is_congested) in enumerate(zip(self.df['system_time'], high_congestion)):
            if is_congested and start_time is None:
                start_time = time
            elif not is_congested and start_time is not None:
                duration = time - start_time
                if duration >= duration_threshold:
                    congestion_periods.append((start_time, time))
                start_time = None
        
        return congestion_periods
    
    def calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate traffic efficiency metrics."""
        total_time = self.df['system_time'].max() - self.df['system_time'].min()
        total_vehicle_hours = self.df['system_total_vehicles'].sum() / 3600  # Convert to hours
        total_delay = self.df['system_total_waiting_time'].sum() / 3600  # Convert to hours
        avg_speed = self.df['system_mean_speed'].mean()
        
        return {
            'total_simulation_time_hours': total_time / 3600,
            'total_vehicle_hours': total_vehicle_hours,
            'total_delay_hours': total_delay,
            'average_speed_ms': avg_speed,
            'delay_ratio': total_delay / total_vehicle_hours if total_vehicle_hours > 0 else 0,
            'throughput_vehicles_per_hour': self.df['system_total_vehicles'].mean()
        }


class ConfigurationManager:
    """Manages different simulation configurations."""
    
    @staticmethod
    def create_rush_hour_config(base_config) -> object:
        """Create configuration for rush hour simulation."""
        from fixed import SimulationConfig
        
        return SimulationConfig(
            simulation_time=base_config.simulation_time,
            net_file=base_config.net_file,
            route_file=base_config.route_file.replace('.rou.xml', '_rush.rou.xml'),
            output_file=base_config.output_file.replace('.csv', '_rush.csv'),
            use_gui=base_config.use_gui,
            step_length=base_config.step_length
        )
    
    @staticmethod
    def create_night_time_config(base_config) -> object:
        """Create configuration for night time simulation."""
        from fixed import SimulationConfig
        
        return SimulationConfig(
            simulation_time=base_config.simulation_time,
            net_file=base_config.net_file,
            route_file=base_config.route_file.replace('.rou.xml', '_night.rou.xml'),
            output_file=base_config.output_file.replace('.csv', '_night.csv'),
            use_gui=base_config.use_gui,
            step_length=base_config.step_length
        )
    
    @staticmethod
    def batch_configurations(base_configs: List) -> List:
        """Create batch configurations for multiple runs."""
        batch_configs = []
        
        for i, config in enumerate(base_configs):
            # Create variations
            for sim_time in [1000, 2000, 3000]:
                new_config = config
                new_config.simulation_time = sim_time
                new_config.output_file = f"batch_run_{i}_{sim_time}s.csv"
                batch_configs.append(new_config)
        
        return batch_configs


class ResultsComparator:
    """Compare results from multiple simulation runs."""
    
    def __init__(self):
        self.results_collection = {}
    
    def add_results(self, name: str, results_df: pd.DataFrame) -> None:
        """Add simulation results with a name for comparison."""
        self.results_collection[name] = results_df
    
    def load_results_from_files(self, file_paths: Dict[str, str]) -> None:
        """Load results from CSV files."""
        for name, file_path in file_paths.items():
            self.results_collection[name] = pd.read_csv(file_path)
    
    def compare_metrics(self, metric: str) -> pd.DataFrame:
        """Compare a specific metric across all results."""
        comparison_data = {}
        
        for name, df in self.results_collection.items():
            if metric in df.columns:
                comparison_data[name] = df[metric]
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison(self, 
                       metrics: List[str], 
                       save_path: Optional[str] = None) -> None:
        """Plot comparison of multiple metrics across simulations."""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 5 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            comparison_df = self.compare_metrics(metric)
            
            for name in comparison_df.columns:
                # Use time from first simulation as x-axis
                time_col = list(self.results_collection.values())[0]['system_time']
                axes[i].plot(time_col[:len(comparison_df[name])], 
                           comparison_df[name], 
                           label=name, 
                           alpha=0.8)
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_xlabel('Simulation Time (s)')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_summary_report(self) -> Dict[str, Dict]:
        """Generate a summary report comparing all simulations."""
        report = {}
        
        for name, df in self.results_collection.items():
            analyzer = SimulationAnalyzer(df)
            report[name] = {
                'summary_stats': analyzer.get_summary_statistics(),
                'efficiency_metrics': analyzer.calculate_efficiency_metrics(),
                'congestion_periods': analyzer.detect_congestion_periods()
            }
        
        return report


def create_batch_runner():
    """Create a batch runner for multiple simulations."""
    from fixed import FixedTimingSimulation
    
    def run_batch_simulations(configs: List, progress_callback=None):
        """Run multiple simulations in batch."""
        results = {}
        
        for i, config in enumerate(configs):
            if progress_callback:
                progress_callback(i, len(configs), config.output_file)
            
            simulation = FixedTimingSimulation(config)
            results[config.output_file] = simulation.run_simulation()
        
        return results
    
    return run_batch_simulations


def export_results_to_formats(df: pd.DataFrame, base_filename: str) -> None:
    """Export results to multiple formats."""
    base_path = Path(base_filename).stem
    
    # CSV (already default)
    df.to_csv(f"{base_path}.csv", index=False)
    
    # Excel
    try:
        df.to_excel(f"{base_path}.xlsx", index=False)
        print(f"Excel file saved: {base_path}.xlsx")
    except ImportError:
        print("openpyxl not installed, skipping Excel export")
    
    # JSON
    df.to_json(f"{base_path}.json", orient='records', indent=2)
    print(f"JSON file saved: {base_path}.json")
    
    # Parquet (efficient for large datasets)
    try:
        df.to_parquet(f"{base_path}.parquet", index=False)
        print(f"Parquet file saved: {base_path}.parquet")
    except ImportError:
        print("pyarrow not installed, skipping Parquet export")


def validate_simulation_setup(config) -> List[str]:
    """Validate simulation configuration and files."""
    issues = []
    
    # Check if network file exists
    if not Path(config.net_file).exists():
        issues.append(f"Network file not found: {config.net_file}")
    
    # Check if route file exists
    if not Path(config.route_file).exists():
        issues.append(f"Route file not found: {config.route_file}")
    
    # Check simulation parameters
    if config.simulation_time <= 0:
        issues.append("Simulation time must be positive")
    
    if config.step_length <= 0:
        issues.append("Step length must be positive")
    
    # Check SUMO environment
    import os
    if 'SUMO_HOME' not in os.environ:
        issues.append("SUMO_HOME environment variable not set")
    
    return issues
