import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the folder containing PNG files
folder_path = "./crnn4_relu/visualization/LP"

# Get a list of all PNG files in the folder
png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]

# Randomly select 30 PNG files
selected_files = random.sample(png_files, 30)

# Set up the figure and axes
fig, axs = plt.subplots(5, 6, figsize=(24, 10))

# Iterate over selected files and plot them
for i, file_name in enumerate(selected_files):
    # Calculate row and column indices
    row = i // 6
    col = i % 6
    
    # Load the image
    img = mpimg.imread(os.path.join(folder_path, file_name))
    
    # Plot the image
    axs[row, col].imshow(img)
    axs[row, col].axis('off')

# Adjust layout
plt.tight_layout()

# Save the diagram
plt.savefig("crnn4_relu.png")

# Show the diagram
plt.show()
