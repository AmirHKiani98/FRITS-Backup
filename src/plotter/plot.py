import os
from glob import glob
import re
import math
from collections import defaultdict
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

palette = list(mcolors.TABLEAU_COLORS.values())
class Plotter:
    def __init__(
        self, 
        main_path: str, 
        col_of_interest: str = "system_total_stopped", 
        cmap: str = "viridis", 
        fixed_path: str = "./output_modification/fixed/4x4/4x4c2c1.csv", 
        baseline_models_path: dict = {}, 
        target_time: int = 600,
        ignore: list = []
    ):
        self.main_path = main_path
        if not isinstance(self.main_path, str):
            raise TypeError("Path must be a string.")

        self.fixed_path = fixed_path
        self.baseline_models_path = baseline_models_path
        self.dataframes = {}
        self.baseline_models_df = {}
        self.col_of_interest = col_of_interest
        self.color_map = plt.get_cmap(cmap)
        self.min_value = float('inf')  # Initialize min_value to infinity
        self.max_value = float('-inf')  # Initialize max_value to negative infinity
        self.target_time = target_time
        self.path = {}
        self.ignore = ignore
        self.shape_path(self.main_path)
        self.read_data()
        self.read_baseline_models()
        self.read_fixed_data()

    def shape_path(self, main_path: str):
        """
        Shapes the main path
        """
        self.path = {}
        print(f"Looking for paths matching: {main_path}")  # Debug output
        
        # If main_path is a directory, look for subdirectories
        if os.path.isdir(main_path):
            print(f"'{main_path}' is a directory. Looking for subdirectories...")
            # Look for subdirectories that match the pattern
            subdirs = [d for d in glob(os.path.join(main_path, "*")) if os.path.isdir(d)]
            
            for path in subdirs:
                for value in self.ignore:
                    if value in path:
                        continue
                key = os.path.basename(path)
                self.path[key] = list(glob(path + "/*"))[0]
            
        else:
            # Original logic for glob patterns
            for path in glob(main_path):
                name_list = path.replace("_nu_", "_mu_").split(os.sep)[-1].split("_")
                key = " ".join([f"\\{name}" if i % 2 == 0 else name for i, name in enumerate(name_list)])
                path_to_alphas = list(glob(path))[0]
                if os.path.isdir(path_to_alphas):
                    self.path[key] = path_to_alphas
        


        



    def read_data(self):
        """
        Reads data from the specified path.
        The path should be a dictionary with keys 'fixed', 'str', and 'dict'.
        """
        for key, value in self.path.items():
            if not isinstance(value, str):
                raise TypeError(f"Value for key '{key}' must be a string.")
            
            if os.path.isfile(value):
                df = pl.read_csv(value)
                self.dataframes[key] = df

            elif os.path.isdir(value):
                files = list(glob(os.path.join(value, "*.csv")))
                self.dataframes[key] = defaultdict(lambda: pl.DataFrame())
                for file in files:
                    alpha_number_search = re.search(r'alpha_(\d+)', file)
                    if alpha_number_search:
                        alpha_number = int(alpha_number_search.group(1))
                        df = pl.read_csv(file)
                        _max = df[self.col_of_interest].max()
                        min_value = df[self.col_of_interest].min()
                        if min_value is not None and isinstance(min_value, (int, float)) and min_value < self.min_value:
                            self.min_value = min_value
                        
                        if _max is not None and isinstance(_max, (int, float)) and _max > self.max_value:
                            self.max_value = _max
                        
                        self.dataframes[key][alpha_number] = pl.concat([self.dataframes[key][alpha_number], df])
                    else:
                        print(f"Warning: File '{file}' does not contain 'alpha' in its name.")
            else:
                print(f"Warning: Path '{value}' is neither a file nor a directory.")

    def read_baseline_models(self):
        """
        Reads baseline models data from the specified path.
        The path should be a dictionary with keys representing model names and values as paths.
        """
        for key, value in self.baseline_models_path.items():
            if not os.path.isdir(value):
                raise ValueError(f"Path for baseline model '{key}' is not a directory: {value}")
            files = list(glob(os.path.join(value, "*.csv")))
            self.baseline_models_df[key] = defaultdict(lambda: pl.DataFrame())
            for file in files:
                alpha_number_search = re.search(r'alpha_(\d+)', file)
                if alpha_number_search:
                    alpha_number = int(alpha_number_search.group(1))
                    df = pl.read_csv(file)
                    _max = df[self.col_of_interest].max()
                    min_value = df[self.col_of_interest].min()
                    if min_value is not None and isinstance(min_value, (int, float)) and min_value < self.min_value:
                        self.min_value = min_value
                    
                    if _max is not None and isinstance(_max, (int, float)) and _max > self.max_value:
                        self.max_value = _max
                    
                    self.baseline_models_df[key][alpha_number] = pl.concat([self.baseline_models_df[key][alpha_number], df])
                else:
                    raise ValueError(f"File '{file}' does not contain 'alpha' in its name.")   

    def read_fixed_data(self):
        """
        Reads fixed data from the specified path.
        """
        if not os.path.isfile(self.fixed_path):
            raise FileNotFoundError(f"Fixed file '{self.fixed_path}' does not exist.")
        df = pl.read_csv(self.fixed_path)
        self.fixed_df = df

    def plot_alpha_specific(self):
        """
        Each plot is related to a specific alpha value.
        """
        keys = [key for key in self.dataframes.keys() if key != 'fixed']
        for key in keys:
            
            n_plots = len(self.dataframes[key])
            if n_plots == 0:
                print(f"Warning: No data found for key '{key}'. Skipping plot.")
                continue
                
            ncols = 2
            nrows = math.ceil(n_plots / ncols)

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), constrained_layout=True)
            axs = axs.flatten()
            digits = re.findall(r'\d+(\.\d+)?', key)
            if digits[1] == "":
                key_to_put = key
            else:
                value = round(float(digits[1]), 2)
                key_to_put = value
            fig.suptitle(rf"$\mu$ = {key_to_put}", fontsize=16)
            self.dataframes[key] = dict(sorted(self.dataframes[key].items(), key=lambda item: item[0]))  # Sort by alpha value
            for idx, (alpha, df) in enumerate(self.dataframes[key].items()):
                if not isinstance(df, pl.DataFrame):
                    raise TypeError(f"Data for key '{key}' and alpha '{alpha}' must be a Polars DataFrame.")
                df = df.sort("system_time")
 
                new_df = df.group_by("system_time").agg(
                    [
                        pl.col(self.col_of_interest).quantile(0.9975).alias("max"),
                        pl.col(self.col_of_interest).quantile(0.0025).alias("min"),
                        pl.col(self.col_of_interest).mean().alias("mean")
                    ]
                )
                    
                
                
                colors = palette[idx % len(palette)]  # Generate a specific color based on alpha index
                axs[idx].plot(new_df["system_time"], new_df["mean"], label="Mean", color=colors)
                axs[idx].fill_between(
                    new_df["system_time"],
                    new_df["min"],
                    new_df["max"],
                    color=colors,
                    alpha=0.2,
                    label="Quartiels 0.0025 - 0.9975"
                )

                # Fixed
                df_fixed = self.fixed_df.clone()
                max_time = new_df["system_time"].max()
                if not isinstance(df_fixed, pl.DataFrame):
                    raise TypeError("Data for 'fixed' must be a Polars DataFrame.")
                df_fixed = df_fixed.filter(pl.col("system_time") <= max_time)
                df_fixed = df_fixed.sort("system_time")
                axs[idx].plot(df_fixed["system_time"], df_fixed[self.col_of_interest], label="Fixed", color='grey', linestyle='--')
                axs[idx].set_ylim(self.min_value, self.max_value)
                # Baseline Models
                for model_name, model_df in self.baseline_models_df.items():
                    if alpha in model_df:
                        model_data = model_df[alpha]
                        if not isinstance(model_data, pl.DataFrame):
                            raise TypeError(f"Data for model '{model_name}' and alpha '{alpha}' must be a Polars DataFrame.")
                        model_data = model_data.sort("system_time")
                        masked_model_data = model_data.filter(pl.col("system_time") <= max_time)
                        grouped = masked_model_data.group_by("system_time").agg(
                            [
                                pl.col(self.col_of_interest).quantile(0.9975).alias("max"),
                                pl.col(self.col_of_interest).quantile(0.0025).alias("min"),
                                pl.col(self.col_of_interest).mean().alias("mean")
                            ]
                        )
                        axs[idx].plot(grouped["system_time"], grouped["mean"], label=f"Mean of {model_name}", linestyle='--')
                        axs[idx].fill_between(
                            grouped["system_time"],
                            grouped["min"],
                            grouped["max"],
                            color="darkred",
                            alpha=0.2,
                            label=f"Quartiles 0.0025 - 0.9975 of {model_name}"
                        )
                axs[idx].set_title(rf"$\alpha$ = {alpha/10}")
                axs[idx].set_xlabel("System Time")
                axs[idx].grid(True)
                axs[idx].set_ylabel(self.col_of_interest)
                if self.target_time != 0:
                    axs[idx].set_xlim(0, self.target_time)
                axs[idx].legend()
            plt.gca().get_yaxis().get_offset_text().set_fontsize(18)
            safe_key = key.replace(" ", "").replace("\\", "").replace(",", "_")
            
            # Create figures directory if it doesn't exist
            os.makedirs('./figures', exist_ok=True)
            
            plt.savefig(f'./figures/{safe_key}_alpha_specific.png', dpi=100, bbox_inches='tight')
            # plt.show()
            plt.close()
    
    def plot_against_each_baseline(self):
        """
        Create separate plots comparing:
        1. FRITS + FedLight vs Fixed
        2. FRITS + FedLight vs Actuated
        """
        keys = [key for key in self.dataframes.keys() if key != 'fixed']
        
        # Create two separate comparisons: vs Fixed and vs Actuated
        comparisons = [
            {"name": "Fixed", "is_file": True},
            {"name": "Actuated", "is_file": False}
        ]
        
        for comparison in comparisons:
            baseline_name = comparison["name"]
            is_file_baseline = comparison["is_file"]
            
            for key in keys:  # For each main model (FRITS variants)
                n_plots = len(self.dataframes[key])
                if n_plots == 0:
                    print(f"Warning: No data found for key '{key}'. Skipping plot.")
                    continue
                    
                ncols = 2
                nrows = math.ceil(n_plots / ncols)

                fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), constrained_layout=True)
                if nrows * ncols == 1:
                    axs = [axs]
                else:
                    axs = axs.flatten()
                    
                # Extract mu value for the plot title
                digits = re.findall(r'\d+(\.\d+)?', key)
                if digits and len(digits) > 1:
                    if digits[1] == "":
                        key_to_put = key
                    else:
                        value = round(float(digits[1]), 2)
                        key_to_put = value
                else:
                    key_to_put = key
                    
                fig.suptitle(f"FRITS (μ = {key_to_put}) and FedLight vs {baseline_name}", fontsize=16)
                
                self.dataframes[key] = dict(sorted(self.dataframes[key].items(), key=lambda item: item[0]))
                
                for idx, (alpha, df) in enumerate(self.dataframes[key].items()):
                    if not isinstance(df, pl.DataFrame):
                        raise TypeError(f"Data for key '{key}' and alpha '{alpha}' must be a Polars DataFrame.")
                    df = df.sort("system_time")
    
                    # Process main model (FRITS)
                    new_df = df.group_by("system_time").agg([
                        pl.col(self.col_of_interest).quantile(0.9975).alias("max"),
                        pl.col(self.col_of_interest).quantile(0.0025).alias("min"),
                        pl.col(self.col_of_interest).mean().alias("mean")
                    ])
                    
                    # Plot main model (FRITS)
                    colors = palette[0]  # First color for FRITS
                    axs[idx].plot(new_df["system_time"], new_df["mean"], label="FRITS", color=colors)
                    axs[idx].fill_between(
                        new_df["system_time"],
                        new_df["min"],
                        new_df["max"],
                        color=colors,
                        alpha=0.2,
                    )

                    max_time = new_df["system_time"].max()
                    
                    # Add FedLight
                    fedlight_model_name = "FedLight-1 and 5"  # Use your FedLight model name here
                    if fedlight_model_name in self.baseline_models_df and alpha in self.baseline_models_df[fedlight_model_name]:
                        fedlight_data = self.baseline_models_df[fedlight_model_name][alpha]
                        if isinstance(fedlight_data, pl.DataFrame):
                            fedlight_data = fedlight_data.sort("system_time")
                            masked_fedlight_data = fedlight_data.filter(pl.col("system_time") <= max_time)
                            grouped_fedlight = masked_fedlight_data.group_by("system_time").agg([
                                pl.col(self.col_of_interest).quantile(0.9975).alias("max"),
                                pl.col(self.col_of_interest).quantile(0.0025).alias("min"),
                                pl.col(self.col_of_interest).mean().alias("mean")
                            ])
                            axs[idx].plot(
                                grouped_fedlight["system_time"], 
                                grouped_fedlight["mean"], 
                                label="FedLight", 
                                color=palette[1],  # Second color for FedLight
                                linestyle='-'
                            )
                            axs[idx].fill_between(
                                grouped_fedlight["system_time"],
                                grouped_fedlight["min"],
                                grouped_fedlight["max"],
                                color=palette[1],
                                alpha=0.2,
                            )
                    
                    # Add the baseline (either Fixed or Actuated)
                    if is_file_baseline:
                        # Fixed baseline (from fixed file)
                        df_fixed = self.fixed_df.clone()
                        if isinstance(df_fixed, pl.DataFrame):
                            df_fixed = df_fixed.filter(pl.col("system_time") <= max_time)
                            df_fixed = df_fixed.sort("system_time")
                            axs[idx].plot(
                                df_fixed["system_time"], 
                                df_fixed[self.col_of_interest], 
                                label="Fixed", 
                                color=palette[2],  # Third color for fixed
                                linestyle='-'
                            )
                    else:
                        # Actuated baseline (from baseline_models_df)
                        actuated_model_name = "Actuated"
                        if actuated_model_name in self.baseline_models_df and alpha in self.baseline_models_df[actuated_model_name]:
                            actuated_data = self.baseline_models_df[actuated_model_name][alpha]
                            if isinstance(actuated_data, pl.DataFrame):
                                actuated_data = actuated_data.sort("system_time")
                                masked_actuated_data = actuated_data.filter(pl.col("system_time") <= max_time)
                                grouped_actuated = masked_actuated_data.group_by("system_time").agg([
                                    pl.col(self.col_of_interest).quantile(0.9975).alias("max"),
                                    pl.col(self.col_of_interest).quantile(0.0025).alias("min"),
                                    pl.col(self.col_of_interest).mean().alias("mean")
                                ])
                                axs[idx].plot(
                                    grouped_actuated["system_time"], 
                                    grouped_actuated["mean"], 
                                    label="Actuated", 
                                    color=palette[2],  # Third color for actuated
                                    linestyle='-'
                                )
                                axs[idx].fill_between(
                                    grouped_actuated["system_time"],
                                    grouped_actuated["min"],
                                    grouped_actuated["max"],
                                    color=palette[2],
                                    alpha=0.2,
                                )
                    
                    # Set y-axis limits
                    axs[idx].set_ylim(self.min_value, self.max_value)
                    
                    # Formatting
                    axs[idx].set_title(rf"$\alpha$ = {alpha/10}")
                    axs[idx].set_xlabel("System Time")
                    axs[idx].grid(True)
                    axs[idx].set_ylabel(self.col_of_interest)
                    if self.target_time != 0:
                        axs[idx].set_xlim(0, self.target_time)
                    axs[idx].legend()
                
                # Save figure
                plt.gca().get_yaxis().get_offset_text().set_fontsize(18)
                safe_key = key.replace(" ", "").replace("\\", "").replace(",", "_")
                safe_baseline_name = baseline_name.replace(" ", "_").replace("-", "_")
                
                # Create figures directory if it doesn't exist
                os.makedirs('./figures', exist_ok=True)
                
                plt.savefig(f'./figures/{safe_key}_and_FedLight_vs_{safe_baseline_name}.png', dpi=100, bbox_inches='tight')
                plt.close()


    def plot_alpha_all(self):
        """
        Each plot contains all alpha values.
        """
        pass



if __name__ == "__main__":
    path = "./src/output_modification/4x4"
    ignore = ["attacked"]
    baseline_models_path = {
        "FedLight-1 and 5": "./src/models/fedlight/output/1_5",
        "Actuated": "./src/models/actuated/output",
        # "FedLight-6 and 11": "./src/models/fedlight/output/6_11",
        # "Fedlight-10": "./src/models/fedlight/output/i4-fedlight",
    }
    
    plotter = Plotter(path, baseline_models_path=baseline_models_path, ignore=ignore)
    #plotter.plot_alpha_specific()  # Original plots with all models
    plotter.plot_against_each_baseline()  # New pairwise comparison plots
    # plotter.plot_alpha_all()  # Uncomment to plot all alpha values in one plot