import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

print("Testing basic functionality...")
sys.stdout.flush()

# Generate some simple data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple plot
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title("Test Plot")
plt.savefig("test_plot.png")

print("Test plot saved to test_plot.png")
sys.stdout.flush()
print("Test completed successfully")
sys.stdout.flush()
