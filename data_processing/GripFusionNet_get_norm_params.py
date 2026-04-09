import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


def compute_mean_std_digit_images(folder_path, resize=(320, 240)):
    """
    Compute mean & std for digit images (*_digit.png).
    Images are resized before computing statistics.

    Args:
        folder_path (str): path to dataset split (e.g. train/)
        resize (tuple): size to resize images (W,H)

    Returns:
        (mean, std): list of floats per RGB channel
    """
    image_paths = glob(os.path.join(folder_path, "*_digit.png"))
    if not image_paths:
        raise ValueError("No digit images found in the specified folder.")

    transform = transforms.Compose([
        transforms.Resize(resize),  # resize for consistency
        transforms.ToTensor()       # convert to tensor [0,1]
    ])

    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    total_pixels = 0

    for path in tqdm(image_paths, desc="Computing digit image stats"):
        img = Image.open(path).convert("RGB")
        tensor = transform(img)  # [3, H, W]
        sum_ += tensor.sum(dim=(1, 2))
        sum_sq += (tensor ** 2).sum(dim=(1, 2))
        total_pixels += tensor.shape[1] * tensor.shape[2]

    mean = sum_ / total_pixels
    std = (sum_sq / total_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def compute_mean_std_rgb_images_no_crop(folder_path):
    """
    Compute mean & std for RGB images (excluding *_digit.png).
    Statistics are computed over the full RGB image.

    Args:
        folder_path (str): path to dataset split (e.g. train/)

    Returns:
        (mean, std): list of floats per RGB channel
    """
    image_paths = glob(os.path.join(folder_path, "*.png"))
    image_paths = [p for p in image_paths if "_digit" not in p]
    if not image_paths:
        raise ValueError("No RGB images found in the specified folder.")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    total_pixels = 0

    for path in tqdm(image_paths, desc="Computing RGB (cropped) image stats"):
        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        sum_ += tensor.sum(dim=(1, 2))
        sum_sq += (tensor ** 2).sum(dim=(1, 2))
        total_pixels += tensor.shape[1] * tensor.shape[2]

    mean = sum_ / total_pixels
    std = (sum_sq / total_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def compute_mean_std_rgb_images(folder_path, roi=(1800, 1600, 2250, 3000)):
    """
    Compute mean & std for RGB images (excluding *_digit.png).
    A fixed ROI crop is applied before computing statistics.

    Args:
        folder_path (str): path to dataset split (e.g. train/)
        roi (tuple): crop box (left, upper, right, lower)

    Returns:
        (mean, std): list of floats per RGB channel
    """
    image_paths = glob(os.path.join(folder_path, "*.png"))
    image_paths = [p for p in image_paths if "_digit" not in p]
    if not image_paths:
        raise ValueError("No RGB images found in the specified folder.")

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.crop(roi)),  # crop ROI
        transforms.ToTensor()
    ])

    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    total_pixels = 0

    for path in tqdm(image_paths, desc="Computing RGB (cropped) image stats"):
        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        sum_ += tensor.sum(dim=(1, 2))
        sum_sq += (tensor ** 2).sum(dim=(1, 2))
        total_pixels += tensor.shape[1] * tensor.shape[2]

    mean = sum_ / total_pixels
    std = (sum_sq / total_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()




if __name__ == "__main__":
    folder = "/home/iat/MA_Fabian/Daten/USB_TCP_combined_split/train"

    # Compute stats separately for Digit & RGB
    digit_mean, digit_std = compute_mean_std_digit_images(folder)
    rgb_mean, rgb_std = compute_mean_std_rgb_images(folder)

    print("\n📊 Digit image normalization parameters:")
    print(f"Mean: {digit_mean}")
    print(f"Std:  {digit_std}")

    print("\n📊 RGB (cropped ROI only, no resize) normalization parameters:")
    print(f"Mean: {rgb_mean}")
    print(f"Std:  {rgb_std}")