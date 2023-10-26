import os

from PIL import Image

def convert_tif_to_gif(input_file, output_file):
    try:
        # Open the input .tif file
        with Image.open(input_file) as img:
            # Convert and save as .gif
            img.save(output_file, format="GIF")
        print(f"Conversion successful. Saved as {output_file}")
    except Exception as e:
        print(f"Error: {e}")




def convert_non_black_to_white(input_file, output_file):
    try:
        with Image.open(input_file) as img:
            # Convert to grayscale
            img = img.convert("L")
            # Threshold to make non-black pixels white
            threshold = 1
            img = img.point(lambda p: p > threshold and 255)

            # Convert back to RGB
            img = img.convert("RGB")

            # Save the result as a new file
            img.save(output_file)
        print(f"Conversion successful. Saved as {output_file}")
    except Exception as e:
        print(f"Error: {e}")

# Usage

for i in range (1, 55):
    if i < 10:
        input_file = "IDRiD_0" + str(i) + "_MA.tif"
        output_file = "IDRiD_0" + str(i) + "_mask.gif"
    else:
        input_file = "IDRiD_" + str(i) + "_MA.tif"
        output_file = "IDRiD_" + str(i) + "_mask.gif"

    convert_tif_to_gif(input_file, output_file)
    convert_non_black_to_white(output_file, output_file)