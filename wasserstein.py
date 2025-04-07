import numpy as np
from PIL import Image
import os
import ot  # POT library

def load_and_prepare_image(path, size=(64, 64)):
    img = Image.open(path).convert('L').resize(size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float64)
    arr /= arr.sum()
    return arr

def save_image(array, path):
    array = array / array.max() * 255
    img = Image.fromarray(array.astype(np.uint8))
    img.save(path)

def wasserstein_interpolate(mu, nu, coords, t):
    mu_flat = mu.ravel()
    nu_flat = nu.ravel()
    
    cost_matrix = ot.dist(coords, coords, metric='euclidean') ** 2
    cost_matrix /= cost_matrix.max()
    
    gamma = ot.emd(mu_flat, nu_flat, cost_matrix)
    
    transported_coords = (1 - t) * coords[:, None, :] + t * coords[None, :, :]
    mass = gamma[..., None] * transported_coords
    interp_coords = mass.sum(axis=1)
    
    nx, ny = mu.shape
    interp_img = np.zeros_like(mu)
    for val, pos in zip(mu_flat, interp_coords):
        i, j = np.round(pos).astype(int)
        if 0 <= i < nx and 0 <= j < ny:
            interp_img[i, j] += val
    
    return interp_img

def generate_wasserstein_frames(img1_path, img2_path, num_frames=20, output_dir='wasserstein_frames', gif_path='wasserstein.gif', size=(64, 64), duration=100):
    os.makedirs(output_dir, exist_ok=True)
    
    mu = load_and_prepare_image(img1_path, size)
    nu = load_and_prepare_image(img2_path, size)
    
    nx, ny = mu.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    coords = np.stack([x.ravel(), y.ravel()], axis=1)

    frame_paths = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        print(f"Generating frame {i+1}/{num_frames} (t={t:.2f})")
        interpolated = wasserstein_interpolate(mu, nu, coords, t)
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        save_image(interpolated, frame_path)
        frame_paths.append(frame_path)

    # Create a GIF
    frames = [Image.open(fp) for fp in frame_paths]
    gif_out_path = os.path.join(output_dir, gif_path)
    frames[0].save(gif_out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"\n✅ GIF saved to: {gif_out_path}")

# Example usage
if __name__ == "__main__":
    generate_wasserstein_frames("source/cavs.webp", "source/lakers.webp", num_frames=30)