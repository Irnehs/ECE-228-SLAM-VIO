import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("train loss.csv")
df2 = pd.read_csv("val loss.csv")


# Extract only the 'Value' column
values = df["Value"]
values2 = df2["Value"]
values = values[:60]
values2 = values2[:60]

# Plot
plt.figure(figsize=(8, 4))
plt.plot(values, linestyle="-", markersize=3)
plt.plot(values2, linestyle="-", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss over 60 Epochs")
plt.legend(['Training', 'Validation'])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()