import torch
from torchvision import transforms
from PIL import Image

def resize_fix(image, size=518):
    """
    Resize image while maintaining aspect ratio and padding to make it square.
    
    Args:
        image (PIL.Image): Input image
        size (int): Target size for both width and height
    
    Returns:
        PIL.Image: Resized and padded image
    """
    # Calculate aspect ratio
    w, h = image.size
    aspect_ratio = w / h
    
    if aspect_ratio > 1:
        # Image is wider than tall
        new_w = size
        new_h = int(size / aspect_ratio)
    else:
        # Image is taller than wide
        new_h = size
        new_w = int(size * aspect_ratio)
    
    # Resize image
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Create a new square image with padding
    square = Image.new('RGB', (size, size), (0, 0, 0))
    square.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    
    return square

def inverse_resize_fix(image, original_size, size=518):
    """
    Invert the resize_fix operation by removing padding and resizing back to original dimensions.
    
    Args:
        image (PIL.Image): Input image that was processed by resize_fix
        original_size (tuple): Original (height, width) of the image
        size (int): Size used in the original resize_fix operation
    
    Returns:
        PIL.Image: Image restored to original dimensions
    """
    # Calculate the aspect ratio of the original image
    orig_h, orig_w = original_size
    aspect_ratio = orig_w / orig_h
    
    # Calculate the dimensions of the resized image before padding
    if aspect_ratio > 1:
        # Image was wider than tall
        new_w = size
        new_h = int(size / aspect_ratio)
    else:
        # Image was taller than wide
        new_h = size
        new_w = int(size * aspect_ratio)
    
    # Calculate the padding offsets
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    
    # Crop the image to remove padding
    cropped = image.crop((pad_x, pad_y, pad_x + new_w, pad_y + new_h))
    
    # Resize back to original dimensions
    restored = cropped.resize((orig_w, orig_h), Image.BILINEAR)
    
    return restored

def normalize(image):
    """
    Normalize image using ImageNet statistics.
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        torch.Tensor: Normalized image tensor
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image) 