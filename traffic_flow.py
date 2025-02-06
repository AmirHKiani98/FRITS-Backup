import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv("d_1_fixed.csv")
df2 = pd.read_csv("d_2_fixed.csv")
plt.figure(figsize=(18,10))
plt.plot(df1.index, df1["system_total_vehicles"], label="Off-Peak Hour Traffic")
plt.plot(df2.index, df2["system_total_vehicles"], label="Peak Hour Traffic")
plt.grid(True)
plt.legend(loc='upper left', fontsize=20)
plt.xlabel('Time (s)', fontsize=25)
plt.ylabel('Active Vehicles', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.gca().get_yaxis().get_offset_text().set_size(18)
plt.savefig("active_vehicles.png")