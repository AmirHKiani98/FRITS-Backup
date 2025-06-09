import os
from glob import glob
import re
import math
from collections import defaultdict
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, path: dict, col_of_interest: str = "system_total_stopped", cmap: str = "viridis", fixed_path: str = "./output/4x4_fixed.csv", baseline_models_path: dict = {}):
        self.path = path
        if not isinstance(self.path, dict):
            raise TypeError("Path must be a dictionary.")
        

        self.fixed_path = fixed_path
        self.baseline_models_path = baseline_models_path
        self.dataframes = {}
        self.baseline_models_df = {}
        self.col_of_interest = col_of_interest
        self.color_map = plt.cm.get_cmap(cmap)
        self.min_value = float('inf')  # Initialize min_value to infinity
        self.max_value = float('-inf')  # Initialize max_value to negative infinity
        self.read_data()
        self.read_baseline_models()
        self.read_fixed_data()

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
                        print("Length of data for alpha", alpha_number, ":", len(self.dataframes[key][alpha_number]))
                    else:
                        raise ValueError(f"File '{file}' does not contain 'alpha' in its name.")

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
            ncols = 2
            nrows = math.ceil(n_plots / ncols)

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), constrained_layout=True)
            axs = axs.flatten()
            fig.suptitle(rf"${key}$", fontsize=16)
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
                    
                
                
                colors = self.color_map(idx / len(self.dataframes[key]))  # Generate a specific color based on alpha index
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
                axs[idx].plot(df_fixed["system_time"], df_fixed[self.col_of_interest], label="Fixed", color='black', linestyle='--')
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
                        axs[idx].plot(grouped["system_time"], grouped["mean"], label=f"Mean of {model_name}", linestyle='--', color="red")
                        axs[idx].fill_between(
                            grouped["system_time"],
                            grouped["min"],
                            grouped["max"],
                            color="red",
                            alpha=0.2,
                            label=f"Quartiles 0.0025 - 0.9975 of {model_name}"
                        )
                axs[idx].set_title(rf"$\alpha$ = {alpha}")
                axs[idx].set_xlabel("System Time")
                axs[idx].grid(True)
                axs[idx].set_ylabel(self.col_of_interest)
                axs[idx].legend()
            plt.gca().get_yaxis().get_offset_text().set_fontsize(18)
            plt.show()


    def plot_alpha_all(self):
        """
        Each plot contains all alpha values.
        """
        pass



if __name__ == "__main__":
    path = {
        "fixed": "./output/4x4_fixed.csv",
        r'\nu = 0.4': "./output/i4-cyber_attack/rl/without_frl/attacked/off-peak/nu_0.4",
        
    }

    baseline_models_path = {
        "FedLight": "./src/models/fedlight/output/i4-fedlight",
    }
    
    plotter = Plotter(path, baseline_models_path=baseline_models_path)
    plotter.plot_alpha_specific()
    # plotter.plot_alpha_all()  # Uncomment to plot all alpha values in one plot