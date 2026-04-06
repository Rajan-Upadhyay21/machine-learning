import numpy as np
import matplotlib.pyplot as plt

population = np.random.exponential(scale=2, size=10000)
sample_means = []

for _ in range(1000):
    sample = np.random.choice(population, size=30)
    sample_means.append(np.mean(sample))

plt.figure(figsize=(8, 5))
plt.hist(sample_means, bins=30)
plt.title("Central Limit Theorem Demo")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
