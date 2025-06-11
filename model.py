import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        resnet = models.resnet34(weights=weights)

        # Replace first conv for 1-channel input
        if in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu
            )
        else:
            self.layer0 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu
            )

        self.pool0  = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.up4 = up_block(512, 256)
        self.up3 = up_block(256, 128)
        self.up2 = up_block(128, 64)
        self.up1 = up_block(64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # ---- Pad input to multiple of 32 ----
        b, c, h, w = x.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        x = F.pad(x, (0, pad_w, 0, pad_h))

        # ---- Encoder ----
        x0 = self.layer0(x)       # ↓2
        x1 = self.pool0(x0)       # ↓4
        x1 = self.layer1(x1)      # ↓4
        x2 = self.layer2(x1)      # ↓8
        x3 = self.layer3(x2)      # ↓16
        x4 = self.layer4(x3)      # ↓32

        # ---- Decoder with skip-adds ----
        d4 = self.up4(x4)         # ↑16
        d4 = d4 + x3
        d3 = self.up3(d4)         # ↑8
        d3 = d3 + x2
        d2 = self.up2(d3)         # ↑4
        d2 = d2 + x1
        d1 = self.up1(d2)         # ↑2
        d1 = d1 + x0

        u0 = self.up0(d1)         # ↑1
        out = torch.sigmoid(self.final(u0))

        # ---- Crop back to original size ----
        return out[:, :, :h, :w]

    def show_first_layer_filters(self, max_filters=32):
        """
        Display convolutional filters from the first layer of a ResNetUNet model.

        Parameters
        ----------
        max_filters : int, optional
            Maximum number of filters to display. Default is 32.
        """

        first_conv = self.layer0[0]  # Should be nn.Conv2d

        if not isinstance(first_conv, nn.Conv2d):
            raise TypeError("First layer is not a Conv2d.")

        # Get weight tensor: shape (out_channels, in_channels, H, W)
        filters = first_conv.weight.data.clone().cpu()

        # Normalize to [0, 1] for visualization
        filters -= filters.min()
        filters /= filters.max()

        out_channels, in_channels, h, w = filters.shape
        n_show = min(max_filters, out_channels)

        # Show only the first input channel for each filter
        plt.figure(figsize=(12, 6))
        for i in range(n_show):
            plt.subplot(4, (n_show + 3) // 4, i + 1)
            plt.imshow(filters[i, 0].numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f'Filter {i}')
        plt.tight_layout()
        plt.show()

    def show_first_layer_outputs(self, image, n_channels=8, device=None):
        """
        Displays the first n output feature maps from the first convolutional layer
        of a segmentation model given a 2D input image.

        Parameters
        ----------
        image : np.ndarray
            2D NumPy array of shape (H, W), dtype uint8 or float32. Pixel values should be in [0, 255] or [0, 1].
        n_channels : int, optional
            Number of output channels to display. Default is 8.
        device : torch.device, optional
            The device to use. If None, automatically detected.

        Returns
        -------
        None
            Displays the output feature maps using matplotlib.
        """
        # Auto-select device
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.to(device)
        self.eval()

        # Normalize image if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert to tensor and add batch & channel dims → [1, 1, H, W]
        img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            features = self.layer0(img_tensor)  # Output shape: [1, C, H, W]
            features = features.squeeze(0).cpu()  # [C, H, W]

        # Clamp to available channels
        n_total = features.shape[0]
        n_show = min(n_channels, n_total)

        # Plot
        plt.figure(figsize=(n_show, 3))
        for i in range(n_show):
            plt.subplot(1, n_show, i + 1)
            plt.imshow(features[i], cmap='gray')
            plt.title(f'Channel {i}')
            plt.axis('off')
        plt.suptitle("First Layer Output", fontsize=16)
        plt.tight_layout()
        plt.show()

    def predict_segmentation(self, image, device=None, output_mode='binary', threshold=0.5):
        """
        Predict segmentation mask for a single image using a trained model.

        Parameters
        ----------
        image : np.ndarray
            2D NumPy array of shape (H, W), dtype uint8, with intensity values in the range [0, 255].
        device : torch.device or None, optional
            Device to run the model on. If None, the device is selected automatically.
        output_mode: str, optional
            The output mode. Can be 'binary' or 'continuous'. Default is 'binary'.
        threshold: float, optional
            Threshold for binary segmentation. Default is 0.5.

        Returns
        -------
        np.ndarray
            2D NumPy array of the predicted mask with shape (H, W), containing values in the range [0, 1].
        """

        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.eval()
        self.to(device)

        # Preprocess image
        img = image.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

        # Save original size
        _, _, h, w = img.shape

        # Pad to multiple of 32 (for ResNet-UNet compatibility)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        img_padded = F.pad(img, (0, pad_w, 0, pad_h))  # Pad right and bottom

        # Inference
        with torch.no_grad():
            pred = self(img_padded)
            pred = pred[:, :, :h, :w]  # Crop to original size
            pred = pred.squeeze().cpu().numpy()  # [H, W]
            if output_mode == 'binary':
                pred = (pred > threshold).astype(np.float32)

        return pred  # Values in [0, 1]
