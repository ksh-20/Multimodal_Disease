"""
gradcam.py
==========
Generic Grad-CAM implementation for any CNN/ResNet model.

Based on: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization," ICCV 2017.
https://arxiv.org/abs/1610.02391

Paper equation reference:
  alpha_k^c = (1/Z) * SUM_{i,j} [ d(y^c) / d(A^k_{ij}) ]     (Eq. 1)
  L^c_GradCAM = ReLU( SUM_k alpha_k^c * A^k )                  (Eq. 2)

where:
  - y^c      : score for class c (before softmax)
  - A^k_{ij} : activation of unit k at spatial location (i,j) in the target layer
  - alpha_k^c: neuron importance weight for feature map k w.r.t. class c
  - Z        : number of pixels in the feature map (normalization constant)
"""

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM explainability method for CNN models.

    Usage
    -----
    >>> cam = GradCAM(model, target_layer=model.layer4[-1])
    >>> heatmap = cam(input_tensor, class_idx=None)  # None -> predicted class
    >>> cam.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Parameters
        ----------
        model        : PyTorch model (eval mode recommended)
        target_layer : The conv layer to extract activations/gradients from.
                       E.g., model.layer4[-1] for ResNet18
        """
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor = None
        self._gradients:   torch.Tensor = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: cache feature map activations A^k."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: cache gradients d(y^c)/d(A^k)."""
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Clean up hooks to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(self, input_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single input image.

        Parameters
        ----------
        input_tensor : torch.Tensor, shape (1, C, H, W)
        class_idx    : int or None. If None, uses the predicted class.

        Returns
        -------
        heatmap : np.ndarray, shape (H, W), values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)         # (1, num_classes)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass: only for the target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Compute alpha_k^c: global average pool of gradients (Eq. 1)
        # Shape: (num_channels, H_feat, W_feat)
        gradients   = self._gradients[0]          # (K, h, w)
        activations = self._activations[0]        # (K, h, w)

        # alpha_k = (1/Z) * sum_{i,j} d(y^c)/d(A^k_{ij})
        alpha = gradients.mean(dim=(1, 2), keepdim=True)  # (K, 1, 1)

        # L^c_GradCAM = ReLU(sum_k alpha_k * A^k)  (Eq. 2)
        cam = (alpha * activations).sum(dim=0)    # (h, w)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam  # shape (h, w), to be upsampled to input size by caller


def compute_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor,
                    target_layer: torch.nn.Module,
                    class_idx: int = None,
                    target_size: tuple = None) -> np.ndarray:
    """
    Convenience function: run Grad-CAM and resize to (H, W).

    Parameters
    ----------
    target_size : (H, W) to upsample heatmap to. If None, returns raw feature-map size.
    """
    import cv2

    cam_obj  = GradCAM(model, target_layer)
    heatmap  = cam_obj(input_tensor, class_idx)
    cam_obj.remove_hooks()

    if target_size is not None:
        heatmap = cv2.resize(heatmap, (target_size[1], target_size[0]),
                              interpolation=cv2.INTER_LINEAR)
    return heatmap


def get_target_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    """
    Resolve a layer by name for common architectures.

    Examples
    --------
    >>> layer = get_target_layer(resnet18, "layer4")
    >>> layer = get_target_layer(custom_cnn, "conv4")
    """
    # ResNet-style: access by attribute chain
    parts = layer_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def batch_gradcam(model: torch.nn.Module, dataloader,
                   target_layer: torch.nn.Module,
                   device: torch.device,
                   n_samples: int = 8) -> list[dict]:
    """
    Generate Grad-CAM heatmaps for a batch of samples (for paper figures).

    Returns
    -------
    list of dicts: {image, heatmap, label, pred, class_idx}
    """
    import torchvision.transforms.functional as TF

    results = []
    model.eval()
    cam_obj = GradCAM(model, target_layer)

    for images, labels in dataloader:
        for i in range(min(n_samples - len(results), images.size(0))):
            img_t = images[i:i+1].to(device)
            with torch.no_grad():
                logits = model(img_t)
                pred   = logits.argmax(dim=1).item()

            # Need grad for Grad-CAM — re-run with grad
            heatmap = cam_obj(img_t, class_idx=pred)

            # Undo normalization for display
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            # Rough denorm for ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_np = np.clip(img_np * std + mean, 0, 1)

            results.append({
                "image":     (img_np * 255).astype(np.uint8),
                "heatmap":   heatmap,
                "label":     labels[i].item(),
                "pred":      pred,
                "class_idx": pred,
            })

        if len(results) >= n_samples:
            break

    cam_obj.remove_hooks()
    return results
