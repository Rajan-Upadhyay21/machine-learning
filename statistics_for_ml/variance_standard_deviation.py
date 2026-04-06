import statistics

data = [12, 15, 18, 20, 22, 25, 28]

variance_value = statistics.variance(data)
std_dev_value = statistics.stdev(data)

print("Data:", data)
print("Variance:", variance_value)
print("Standard Deviation:", std_dev_value)
