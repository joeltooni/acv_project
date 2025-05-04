import cv2
import numpy as np
import torch
import os

def convert_depth_to_grayscale(input_path, output_path=None):
    """
    Convert an RGB depth image to grayscale and save it.
    
    Parameters:
    -----------
    input_path : str
        Path to the input RGB depth image
    output_path : str, optional
        Path to save the grayscale image. If None, uses input_path with '_gray' suffix.
    
    Returns:
    --------
    grayscale : numpy.ndarray
        The grayscale depth image
    """
    # Read the image
    img = cv2.imread(input_path)
    print(f"Original image shape: {img.shape}")
    
    # Convert from BGR to RGB (OpenCV reads as BGR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract channels
    R, G, B = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
    
    # Convert to grayscale using the luminance formula
    grayscale = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Convert to appropriate data type
    grayscale = grayscale.astype(np.uint8)
    print(f"Grayscale image shape: {grayscale.shape}")
    
    # Save the grayscale image if output_path is provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_gray{ext}"
    
    cv2.imwrite(output_path, grayscale)
    print(f"Grayscale image saved to: {output_path}")
    
    return grayscale

def convert_to_torch_tensor(grayscale_np):
    """
    Convert numpy grayscale image to PyTorch tensor
    
    Parameters:
    -----------
    grayscale_np : numpy.ndarray
        Grayscale image as numpy array
    
    Returns:
    --------
    tensor : torch.Tensor
        Grayscale image as PyTorch tensor
    """
    # Convert to float
    tensor = torch.from_numpy(grayscale_np.astype(np.float32))
    print(f"PyTorch tensor shape: {tensor.shape}")
    return tensor

if __name__ == "__main__":
    # Replace with your actual depth image path
    depth_rgb_path = "Example/depth.png"
    output_path = "Example/depth_gray.png"
    
    # Convert to grayscale and save
    grayscale = convert_depth_to_grayscale(depth_rgb_path, output_path)
    
    # Convert to torch tensor (if needed for your model)
    depth_tensor = convert_to_torch_tensor(grayscale)
    
    # Save tensor if needed
    torch.save(depth_tensor, "Example/depth_tensor.pt")
    print("PyTorch tensor saved to: Example/depth_tensor.pt")