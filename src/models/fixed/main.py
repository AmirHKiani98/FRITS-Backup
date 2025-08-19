
from src.models.fixed.model import SimulationConfig, FixedTimingSimulation

def main():
    # Initialize necessary components
    simulation_config = SimulationConfig(
        3200,
        net_file="../../networks/4x4.net.xml",
        route_file="../../routes/4x4c2c1.rou.xml",
        output_file="./output_modification/fixed/4x4/4x4c2c1.csv",
        use_gui=False,
        step_length=1,
        addtional_file="../../additionals/4x4.add.xml"
    )
    model = FixedTimingSimulation(simulation_config)
    model.run_simulation()
if __name__ == "__main__":
    main()
