import numpy as np

# Load the .npy file
arr = np.memmap('/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/hellaswag/2048_core-task-trainsets-v3_hellaswag_00000.npy')

# Print shape and total number of tokens
print("Shape:", arr.shape)
print("Total number of tokens:", arr.size)  # or np.prod(arr.shape)