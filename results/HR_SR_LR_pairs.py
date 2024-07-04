import os
import matplotlib.pyplot as plt
from PIL import Image

# Paths to HR, SR, and LR folders
hr_folder = "./HR"
sr_folder = "./CRNN_232730_40k/visualization/LP"
lr_folder = "./LR_bicubic"

# Output folder to save diagrams
output_folder = "HR_SR_LR_diagram"
os.makedirs(output_folder, exist_ok=True)

# Filenames of the HR and LR images
# filenames = ["000075", "000120", "000283", "000318", "000487"]
# filenames = ["000642", "000703", "000791", "000961", "001069"]
# filenames = ["000897", "000951", "000974", "001000", "001079"]

# 10 pairs good result
# filenames = ["000487", "000791", "000961", "001069", "000091"]
filenames = ["000771", "000781", "000855", "000867", "001079"]

# Create figure and axes
fig, axs = plt.subplots(3, len(filenames), figsize=(20, 8))

for idx, filename in enumerate(filenames):
    # HR image
    hr_filename = filename + ".png"
    hr_image = Image.open(os.path.join(hr_folder, hr_filename))
    axs[0, idx].imshow(hr_image)
    #axs[0, idx].set_title("HR", fontsize=10)
    axs[0, idx].axis("off")

    # SR image
    sr_filename = filename + "_SR.png"
    sr_image = Image.open(os.path.join(sr_folder, sr_filename))
    axs[1, idx].imshow(sr_image)
    #axs[1, idx].set_title("SR", fontsize=10)
    axs[1, idx].axis("off")

    # LR image
    lr_filename = filename + ".png"
    lr_image = Image.open(os.path.join(lr_folder, lr_filename))
    axs[2, idx].imshow(lr_image)
    #axs[2, idx].set_title("LR", fontsize=10)
    axs[2, idx].axis("off")

plt.tight_layout()
output_path = os.path.join(output_folder, "CRNN_232730_40k.png")
plt.savefig(output_path)
plt.show()
