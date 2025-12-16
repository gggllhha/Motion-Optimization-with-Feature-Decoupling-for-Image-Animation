import os

from PIL import Image


def crop_image_to_tiles(input_image_path, output_dir):
    """
    Crop an N*256 resolution .png image into 256*256 tiles and save them.

    Args:
        input_image_path (str): Path to the input .png image.
        output_dir (str): Path to the folder where cropped tiles will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Open the image
        with Image.open(input_image_path) as img:
            width, height = img.size

            if height != 256:
                raise ValueError("Input image height must be 256 pixels.")

            num_tiles = width // 256

            for i in range(num_tiles):
                # Define the crop box (left, upper, right, lower)
                left = i * 256
                upper = 0
                right = left + 256
                lower = 256

                # Crop the image
                tile = img.crop((left, upper, right, lower))

                # Save the tile
                tile_filename = f"{i:05d}.png"
                tile_path = os.path.join(output_dir, tile_filename)
                tile.save(tile_path)
                print(f"Saved: {tile_path}")

    except Exception as e:
        print(f"Error processing image: {e}")

input_image = r"D:\wqw\Thin-Plate-Spline-Motion-Model-main\output_occlusion.png"
output_tiles_dir = r"C:\Users\Lenovo\Desktop\fig"
crop_image_to_tiles(input_image, output_tiles_dir)