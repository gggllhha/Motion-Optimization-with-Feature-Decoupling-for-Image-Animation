import os
from PIL import Image


def extract_gif_frames(input_dir, output_dir):
    """
    Extract frames from .gif files in the input directory and save them as .png files in the corresponding folders.

    Args:
        input_dir (str): Path to the folder containing .gif files.
        output_dir (str): Path to the folder where extracted frames will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse all files in the input directory
    for i in range(100):
        gif_filename = f"{i:05d}.gif"
        gif_path = os.path.join(input_dir, gif_filename)

        # Check if the gif file exists
        if not os.path.isfile(gif_path):
            print(f"File not found: {gif_filename}")
            continue

        # Create a corresponding folder for the extracted frames
        frame_output_dir = os.path.join(output_dir, f"{i:05d}")
        os.makedirs(frame_output_dir, exist_ok=True)

        try:
            # Open the GIF file
            with Image.open(gif_path) as gif:
                frame_index = 0

                # Iterate through each frame in the GIF
                while True:
                    # Construct the output file path
                    frame_filename = f"frame_{frame_index:03d}.png"
                    frame_path = os.path.join(frame_output_dir, frame_filename)

                    # Save the current frame as a PNG
                    gif.save(frame_path, format="PNG")
                    print(f"Saved: {frame_path}")

                    frame_index += 1

                    # Move to the next frame
                    gif.seek(frame_index)

        except EOFError:
            # End of frames in the GIF
            print(f"Finished extracting: {gif_filename}")
        except Exception as e:
            print(f"Error processing {gif_filename}: {e}")


# Specify the input and output directories
input_directory = r"D:\wqw\backup\datasets\mgif\test"
output_directory = r"D:\wqw\mgif_gt"

# Run the extraction function
extract_gif_frames(input_directory, output_directory)