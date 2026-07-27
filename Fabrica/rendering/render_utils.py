import os
import numpy as np
import imageio
import math


def get_color(part_ids, normalize=True):
    color_map = {}
    if len(part_ids) <= 2:
        colors = np.array([
            [107, 166, 161, 255],
            [209, 184, 148, 255],
        ], dtype=int)
    else:
        colors = np.array([
            [217, 41, 41, 255],   # red
            [242, 162, 12, 255],  # yellow
            [49, 140, 7, 255],    # green
            [15, 113, 242, 255],  # blue
            [142, 107, 191, 255], # purple
        ], dtype=int)
    if normalize: colors = colors.astype(float) / 255.0
    for i, part_id in enumerate(part_ids):
        color_map[part_id] = colors[i % len(colors)]
    return color_map


def get_origin_camera(
        horizontal_angle: float = 30.0,
        vertical_angle: float = 45.0,
        radius: float = 2.0):

    horizontal_angle = math.radians(horizontal_angle)
    vertical_angle = math.radians(vertical_angle)

    x = radius * math.sin(vertical_angle) * math.cos(horizontal_angle)
    y = radius * math.sin(vertical_angle) * math.sin(horizontal_angle)
    z = radius * math.cos(vertical_angle)
    return np.array((x, y, z))


def images_to_video(image_paths, output_video_path, fps: int = 30):
    # Ensure the list of images is not empty
    if not image_paths:
        raise ValueError("The list of image paths is empty")

    # Read and write the images to a video
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path, pilmode="RGBA")

            rgb = image[:, :, :3]
            alpha = image[:, :, 3]

            # Create a white background
            white_background = np.ones_like(rgb) * 255

            # Composite the RGB image with the white background using the alpha channel
            alpha_normalized = alpha[:, :, None] / 255.0
            image = (rgb * alpha_normalized + white_background * (1 - alpha_normalized)).astype(np.uint8)

            writer.append_data(image)


class SuppressStdout:
    def __init__(self):
        self.old_fd = None
        self.null_fd = None

    def __enter__(self):
        # Save the original stdout file descriptor
        self.old_fd = os.dup(1)
        # Open /dev/null and duplicate it to stdout
        self.null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null_fd, 1)  # Redirect stdout to /dev/null
        return self  # In case you want to return something from the context manager

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original stdout file descriptor
        os.dup2(self.old_fd, 1)
        os.close(self.old_fd)  # Clean up the saved file descriptor
        os.close(self.null_fd)  # Clean up /dev/null descriptor
        self.old_fd = None
        self.null_fd = None  # Clear references for safety
