
from src.models.fixed.model import SimulationConfig, FixedTimingSimulation

def main():
    # Initialize necessary components
    simulation_config = SimulationConfig(
        3200,
        net_file="../../../net/4x4.net.xml",
        route_file="../../../rou/4x4c2c1.rou.xml",
        output_file="./output_modification/fixed/4x4/4x4c2c1.csv",
        use_gui=False,
        step_length=0.1
    )
    model = FixedTimingSimulation(simulation_config)
    model.run_simulation()
if __name__ == "__main__":
    main()
