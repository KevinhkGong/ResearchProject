from PIL import Image

# Create a black image of size 3390x3390
width, height = 3390, 3390
black_image = Image.new('RGB', (width, height), color='black')

# Save the image
black_image.save('black_image.jpg')
