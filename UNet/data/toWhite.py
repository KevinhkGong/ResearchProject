from PIL import Image
import  os
def convert_non_black_to_white(input_file, output_file):
    try:
        with Image.open(input_file) as img:
            # Convert to grayscale
            img = img.convert("L")

            # Threshold to make non-black pixels white
            threshold = 1
            img = img.point(lambda p: p > threshold and 255)

            # Convert back to RGB
            # img = img.convert("RGB")

            # Save the result as a new file
            img.save(output_file)
        print(f"Conversion successful. Saved as {output_file}")
    except Exception as e:
        print(f"Error: {e}")


input_folder = 'output_mask_test'
output_folder = 'output_mask_white_test'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through the files in the input folder
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    convert_non_black_to_white(input_path, output_path)

print("Conversion complete.")
