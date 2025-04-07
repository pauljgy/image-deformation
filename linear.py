from PIL import Image
import numpy as np
import os

def interpolate_images(image1_path, image2_path, alpha, output_path='interpolated_image.png'):
    """
    Linearly interpolates between two images and saves the result.
    """
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same dimensions!")

    interpolated = (1 - alpha) * arr1 + alpha * arr2
    interpolated = np.clip(interpolated, 0, 255).astype(np.uint8)

    result_img = Image.fromarray(interpolated)
    result_img.save(output_path)
    return result_img

def generate_interpolation_sequence(image1_path, image2_path, num_frames, output_dir='interpolation_frames'):
    """
    Generates a sequence of interpolated frames between two images.

    Parameters:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        num_frames (int): Number of frames to generate (including start and end).
        output_dir (str): Directory to save the frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        interpolate_images(image1_path, image2_path, alpha, output_path=frame_path)
        print(f"Saved frame {i+1}/{num_frames}: {frame_path}")

def create_gif_from_frames(frame_dir, output_gif_path='interpolation.gif', duration=100):
    """
    Creates a GIF from a sequence of image frames.

    Parameters:
        frame_dir (str): Directory containing the frames.
        output_gif_path (str): Output GIF file path.
        duration (int): Duration of each frame in milliseconds.
    """
    frames = []
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    
    for file in frame_files:
        frame_path = os.path.join(frame_dir, file)
        frames.append(Image.open(frame_path))

    if frames:
        frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f"GIF saved to {output_gif_path}")
    else:
        print("No frames found to create GIF.")

# Example usage
if __name__ == "__main__":
    frame_dir = "interpolation_frames"
    generate_interpolation_sequence("cavs.webp", "lakers.webp", num_frames=30, output_dir=frame_dir)
    create_gif_from_frames(frame_dir, output_gif_path="interpolation.gif", duration=100)