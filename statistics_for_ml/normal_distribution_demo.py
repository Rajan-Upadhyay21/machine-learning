import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 200)
y = norm.pdf(x, loc=0, scale=1)

plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.title("Normal Distribution")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()
