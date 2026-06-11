import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# cori_week_avg = pd.read_csv("cori_week_avg.csv")
# marconi_week_avg = pd.read_csv("marconi_week_avg.csv")
# perlmutter_week_avg = pd.read_csv("perlmutter_week_avg.csv")
# hawk_week_avg = pd.read_csv("hawk_week_avg.csv")
azure_inference = pd.read_csv("example_inference_azure_conv.csv")

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.2)
sns.lineplot(x="timestamp_hr", y="watts", data=azure_inference, label="Conversation")
plt.xlabel("Time (hours)")
plt.ylabel("Load (p.u.)")
plt.legend()
plt.show()
