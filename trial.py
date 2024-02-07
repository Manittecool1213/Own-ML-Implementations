import numpy as np

# Get information about float64 type
float64_info = np.finfo(np.float64)

# Print the largest possible value
print("Largest possible value for float64:", float64_info.max)
