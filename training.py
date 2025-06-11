import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from model import ResNetUNet
import numpy as np
import os
import albumentations
from helper_functions import plot_prediction_triplets
from tqdm import tqdm

class BoneSegDataset(Dataset):
    def __init__(self, images, masks, augment=False):
        """
        images: list of uint8 ndarrays with values [0…255]
        masks:  list of uint8 ndarrays with values {0,10}
        """
        self.images = images
        self.masks  = masks
        self.augment = augment

        self.transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Affine(scale=(2/3, 3/2), translate_percent=(0.05, 0.05), rotate=(0, 360), shear=(-5, 5), p=1),
            albumentations.ElasticTransform(p=0.2),
            albumentations.RandomBrightnessContrast(p=0.2),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        msk = self.masks[idx].astype(np.float32) / 10.0

        if self.augment:
            augmented = self.transform(image=img, mask=msk)
            img, msk = augmented['image'], augmented['mask']

        img = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]
        msk = torch.from_numpy(msk).unsqueeze(0)  # [1,H,W]
        return img, msk

def pad_collate(batch):
    """
    Pads a batch of (image, mask) pairs to the same height and width and stacks them into tensors.

    This function is intended as a `collate_fn` for PyTorch's DataLoader to handle variable-sized
    inputs by padding each image and mask in the batch to the maximum height and width found
    in that batch.

    Parameters
    ----------
    batch : list of tuple of torch.Tensor
        A list of (image, mask) pairs. Each image and mask is a 3D tensor of shape
        (C, H, W), where C is the number of channels.

    Returns
    -------
    padded_images : torch.Tensor
        A 4D tensor of shape (B, C, H_max, W_max), where B is the batch size and
        H_max and W_max are the maximum height and width in the batch.

    padded_masks : torch.Tensor
        A 4D tensor of shape (B, C, H_max, W_max), where each mask has been padded
        to match the size of the corresponding image.

    Notes
    -----
    - Padding is applied to the bottom and right sides of the tensors.
    - Assumes all images and masks have the same number of channels.
    """

    images, masks = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded_images, padded_masks = [], []
    for img, msk in zip(images, masks):
        h, w = img.shape[1], img.shape[2]
        pad_h = max_h - h
        pad_w = max_w - w
        img_p = F.pad(img, (0, pad_w, 0, pad_h))
        msk_p = F.pad(msk, (0, pad_w, 0, pad_h))
        padded_images.append(img_p)
        padded_masks.append(msk_p)
    return torch.stack(padded_images), torch.stack(padded_masks)

class DiceLoss(nn.Module):
    def forward(self, predictions, targets, eps=1e-6):
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (predictions * targets).sum()
        return 1 - (2*intersection + eps) / (predictions.sum() + targets.sum() + eps)


def train_unet_model(images, masks, augment=False, n_epochs=100, lr=1e-4, show_test_predictions=True, device_override=None):
    """
    Train a ResNet-UNet model for binary image segmentation and evaluate it on a test set.

    Parameters
    ----------
    images : list of np.ndarray
        List of grayscale input images, each with shape (H, W), dtype uint8, and pixel values in [0, 255].
    masks : list of np.ndarray
        List of corresponding binary segmentation masks, each with shape (H, W), dtype uint8,
        with pixel values in {0, 10}.
    augment : bool, optional
        If True, applies data augmentation during training. Default is False.
    n_epochs : int, optional
        Number of training epochs. Default is 100.
    lr : float, optional
        Learning rate for the Adam optimizer. Default is 1e-4.
    show_test_predictions : bool, optional
        If True, displays prediction results from the test set after training. Default is True.
    device_override : str, optional
        Device override to use. Default is None.

    Returns
    -------
    model : torch.nn.Module
        The trained ResNet-UNet model loaded with weights from the best validation epoch.

    Notes
    -----
    - Uses 30 training samples, 10 validation samples, and 10 test samples by default.
    - Saves model checkpoints to `./checkpoints/` based on best validation loss.
    - Training and validation loss curves are plotted.
    - If `show_test_predictions=True`, visualizes test images, ground truth masks, and predicted masks.
    """

    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    def combined_loss(prediction, target):
        return 0.5 * bce_loss(prediction, target) + 0.5 * dice_loss(prediction, target)

    train_indices = list(range(0, 30))
    val_indices = list(range(30, 40))
    test_indices = list(range(40, 50))

    train_ds = Subset(BoneSegDataset(images, masks, augment=augment), train_indices)
    val_ds = Subset(BoneSegDataset(images, masks, augment=False), val_indices)
    test_ds = Subset(BoneSegDataset(images, masks, augment=False), test_indices)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=10, collate_fn=pad_collate)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device_override is not None:
        device = torch.device(device_override)
    print(f"Using device: {device}")

    model = ResNetUNet(out_channels=1, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch():
        model.train()
        total_loss = 0
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            preds = model(imgs)
            loss = combined_loss(preds, msks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def eval_loss(loader):
        model.eval()
        total = 0
        with torch.no_grad():
            for imgs, msks in loader:
                imgs, msks = imgs.to(device), msks.to(device)
                total += combined_loss(model(imgs), msks).item()
        return total / len(loader)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    ckpt_dir = "../Musterlösung/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    pbar = tqdm(range(n_epochs), total=n_epochs, desc="Training Progress", unit="epoch")

    for epoch in pbar:

        tr_loss = train_epoch()
        val_loss = eval_loss(val_loader)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        #print(f"Epoch {epoch:02d}: Train Loss={tr_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:

            if 'best_ckpt_path' in locals():
                if os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)   

            best_val_loss = val_loss
            best_ckpt_path = os.path.join(ckpt_dir, f"best_epoch{epoch:02d}_valloss{val_loss:.4f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_ckpt_path)
            #print(f"  ↳ New best model saved to {best_ckpt_path}")

        pbar.set_postfix(train_loss=tr_loss, val_loss=val_loss)

    epochs = list(range(1, n_epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    print(f"Best checkpoint was: {best_ckpt_path} with val_loss={best_val_loss:.4f}. Loading checkpoint...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Ensure model is in evaluation mode
    model.eval()

    test_loss = eval_loss(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # Get a batch of test images and masks
    images, masks = next(iter(test_loader))
    images, masks = images.to(device), masks.to(device)

    # Compute predictions
    with torch.no_grad():
        preds = model(images)

    # Move tensors to CPU and convert to numpy
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()

    # Number of images to visualize
    N = images_np.shape[0]

    if show_test_predictions:
        plot_prediction_triplets(images_np, preds_np, masks_np)

    return model


def test_pad_collate():
    # create dummy batch with varying sizes
    img1 = torch.ones(1, 100, 120)
    msk1 = torch.zeros(1, 100, 120)

    img2 = torch.ones(1, 90, 100) * 2
    msk2 = torch.zeros(1, 90, 100) + 10

    batch = [(img1, msk1), (img2, msk2)]
    images, masks = pad_collate(batch)

    # test shapes
    assert images.shape == (2, 1, 100, 120)
    assert masks.shape == (2, 1, 100, 120)

    # check original content is preserved
    assert torch.all(images[0, :, :100, :120] == 1)
    assert torch.all(images[1, :, :90, :100] == 2)

    assert torch.all(masks[0, :, :100, :120] == 0)
    assert torch.all(masks[1, :, :90, :100] == 10)

    # check padded regions are zero
    assert torch.all(images[1, :, 90:, :] == 0)
    assert torch.all(images[1, :, :, 100:] == 0)
    assert torch.all(masks[1, :, 90:, :] == 0)
    assert torch.all(masks[1, :, :, 100:] == 0)

# Run manually (or use `pytest`)
if __name__ == "__main__":
    test_pad_collate()
    print("Test passed.")