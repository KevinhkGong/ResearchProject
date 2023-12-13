from PIL import Image
import os

# Define the input and output folders
input_folder = '2. Haemorrhages Processed_test'
output_folder = 'output_mask_train'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        # Construct the full file paths
        input_path = os.path.join(input_folder, filename)
        output_filename_list = os.path.splitext(filename)[0].split("_")
        output_filename = output_filename_list[0] + "_" + output_filename_list[1] + "_mask.gif"
        print(output_filename)
        output_path = os.path.join(output_folder, output_filename)

        # Open and convert the image
        with Image.open(input_path) as img:
            img.save(output_path, format='GIF')

print("Conversion complete.")
