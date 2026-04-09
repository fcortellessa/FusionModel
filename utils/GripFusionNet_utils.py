import torch
import yaml
import os

def reparameterize(mu, logvar):
    """
    Reparameterization trick: sample from N(mu, sigma^2) using a standard normal variable.
    
    Args:
        mu (torch.Tensor): Mean vector of the latent Gaussian distribution.
        logvar (torch.Tensor): Log-variance vector (log of variance).
    
    Returns:
        torch.Tensor: Sampled latent vector z.
    """
    std = torch.exp(0.5 * logvar)  # Convert logvar to standard deviation
    eps = torch.randn_like(std)    # Sample from standard normal N(0, I)
    return mu + std * eps  # Apply reparameterization trick

# def kl_divergence(tensor1, tensor2):
#     mu1, logvar1 = tensor1[:, :2], tensor1[:, 2:]
#     mu2, logvar2 = tensor2[:, :2], tensor2[:, 2:]

#     sigma1_sq = torch.exp(torch.clamp(logvar1, min=-10, max=10))  # Safe range
#     sigma2_sq = torch.exp(torch.clamp(logvar2, min=-10, max=10))

#     if torch.isnan(sigma1_sq).any() or torch.isnan(sigma2_sq).any():
#         print("🚨 NaN detected in KL computation")
#         return torch.tensor(0.0, device=tensor1.device)

#     kl = 0.5 * (torch.log(sigma2_sq) - torch.log(sigma1_sq)
#                 + sigma1_sq / sigma2_sq
#                 + (mu1 - mu2).pow(2) / sigma2_sq
#                 - 1)

#     return kl.sum()

def kl_divergence_to_gaussian(mu, logvar):
    """
    Compute the KL divergence between the learned Gaussian distribution
    q(y) = N(mu, exp(logvar)) and the standard normal prior p(y) = N(0, 1).

    Args:
        mu (torch.Tensor): Predicted mean values (batch_size, n_dims).
        logvar (torch.Tensor): Predicted log-variance values (batch_size, n_dims).

    Returns:
        torch.Tensor: KL divergence loss.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def kl_divergence(mu1, logvar1, mu2, logvar2):
    """
    Compute the KL divergence between two Gaussian distributions:
    P = N(mu1, exp(logvar1)) and Q = N(mu2, exp(logvar2)).
    
    Args:
        mu1 (torch.Tensor): Mean of the first distribution P.
        logvar1 (torch.Tensor): Log-variance of P.
        mu2 (torch.Tensor): Mean of the second distribution Q.
        logvar2 (torch.Tensor): Log-variance of Q.
    
    Returns:
        torch.Tensor: KL divergence loss.
    """
    var1, var2 = logvar1.exp(), logvar2.exp()
    return 0.5 * torch.sum((var1 / var2) + ((mu2 - mu1).pow(2) / var2) - 1 + (logvar2 - logvar1))


# Load YAML Configuration
def load_config(config_path):
    print(f"🔍 Attempting to load config from: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print(f"✅ Successfully loaded config: {config}")
    return config

def save_checkpoint(epoch, model, optimizer, val_loss, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss  # Save the best validation loss
    }
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved at {path}, with validation loss: {val_loss}")

# Load the most recent checkpoint if available
def load_checkpoint(model, checkpoint_dir, device):
    """
    Loads the 'best_checkpoint.pth' model checkpoint from the given directory,
    including the best validation loss achieved.
    
    Args:
        model (torch.nn.Module): Model to load checkpoint into.
        checkpoint_dir (str): Directory where checkpoints are stored.
        device (torch.device): Device to map model parameters.
    
    Returns:
        model (torch.nn.Module): Model with loaded weights.
        int: Last saved epoch or a default value (returns 0 if no checkpoint is found).
        float: Best validation loss or a large default value (returns float('inf') if no val_loss is found).
    """
    checkpoint_path = checkpoint_dir
    print(f"checkpoint path {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print("⚠️ Checkpoint file not found.")
        return model, 0, float('inf')  # Return infinity if no checkpoint to indicate no best loss available

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        last_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("val_loss", float('inf'))  # Use inf as default if val_loss not found
        print(f"✅ Loaded checkpoint from epoch {last_epoch} with best validation loss: {best_val_loss}")
    else:
        print("⚠️ Checkpoint does not contain 'model_state_dict'. Using raw weights instead.")
        model.load_state_dict(checkpoint)  # Assuming the whole state dict was directly saved
        last_epoch = 0  # If not tracking epochs in this mode
        best_val_loss = float('inf')  # Default to infinity

    return model, last_epoch, best_val_loss

def select_bottom_four(tensor):
    """
    Selects the bottom two entries from the first half and the bottom two entries from the second half
    and returns a new tensor of shape (1,4).
    
    Args:
        tensor (torch.Tensor): A 1D or 2D tensor of shape (1, N), where N is even.
    
    Returns:
        torch.Tensor: A tensor of shape (1,4) containing the selected entries.
    """
    if tensor.dim() != 2 or tensor.shape[0] != 1:
        raise ValueError("Input tensor must have shape (1, N)")

    N = tensor.shape[1]
    if N < 4 or N % 2 != 0:
        raise ValueError("Tensor must have an even number of elements and at least 4 elements.")

    half_N = N // 2

    # Select the last two elements from the first half and second half
    first_half_bottom = tensor[:, half_N-2:half_N]  # Last two elements of first half
    second_half_bottom = tensor[:, N-2:N]  # Last two elements of second half

    # Concatenate along the last dimension
    result = torch.cat((first_half_bottom, second_half_bottom), dim=-1)

    return result


def get_triplets(input_folder):
    """
    Find all triplets (rgb image, digit image, pose json) in the dataset folder.
    Skip samples with timestamps that appear in the excluded list.
    """
    triplets = []
    files = os.listdir(input_folder)
    file_set = set(files)

    for file in files:
        # Look for base RGB image
        if file.endswith('.png') and '_digit' not in file:
            timestamp = file.replace('.png', '')

            rgb_file = f"{timestamp}.png"
            digit_file = f"{timestamp}_digit.png"
            json_file = f"{timestamp}.json"

            # Accept only if all three files exist
            if all(f in file_set for f in [rgb_file, digit_file, json_file]):
                triplets.append((timestamp, rgb_file, digit_file, json_file))
    return triplets
