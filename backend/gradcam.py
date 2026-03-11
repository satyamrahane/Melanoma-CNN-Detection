"""
Grad-CAM — Gradient-weighted Class Activation Mapping
=====================================================
Generates visual heatmaps showing which regions of the dermoscopic image
the EfficientNet-B3 model focused on to make its prediction.

This is the EXPLAINABILITY component of MelanomaAI.
Judges and clinicians can see *where* the model is looking —
confirming it focuses on lesion boundaries, asymmetry, and color variation.

Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017)
"""

import cv2
import numpy as np
import torch


class GradCAM:
    """Grad-CAM for EfficientNet-B3 (targets last convolutional block)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        # Default: last conv block of EfficientNet
        self.target_layer = target_layer or model.features[-1]
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor [1, 3, H, W]

        Returns:
            cam: numpy array [H, W] in range [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        output = self.model(input_tensor)
        
        # To avoid vanishing gradients from Sigmoid when prob is near 0 or 1,
        # we compute Grad-CAM with respect to the pre-sigmoid logit.
        p_val = torch.clamp(output, 1e-7, 1.0 - 1e-7)
        logit = torch.log(p_val / (1.0 - p_val))
        
        # If the model predicts benign (p < 0.5), we want the gradients for the benign class
        if p_val.item() < 0.5:
            logit = -logit
            
        logit.backward()

        gradients = self.gradients.cpu().numpy()[0]     # [C, h, w]
        activations = self.activations.cpu().numpy()[0] # [C, h, w]

        # Global average pool gradients → channel weights
        weights = np.mean(gradients, axis=(1, 2))       # [C]

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — keep only positive influence
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image dimensions
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam


def generate_heatmap_overlay(
    original_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img: RGB image [H, W, 3] uint8
        cam: Grad-CAM output [H, W] float32 [0,1]
        alpha: transparency of heatmap overlay
        colormap: OpenCV colormap constant

    Returns:
        overlay: RGB image [H, W, 3] uint8
    """
    h, w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to uint8 heatmap
    heatmap = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = np.uint8(original_img * (1 - alpha) + heatmap_rgb * alpha)
    return overlay
