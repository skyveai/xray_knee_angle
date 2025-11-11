import torch
import torch.nn.functional as F

def softargmax_2d(heatmaps, beta=100):
    """
    heatmaps: (B, K, H, W)
    beta: controls sharpness of softmax
    Returns: (B, K, 2) keypoints in (x, y)
    """
    B, K, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B, K, -1)
    heatmaps = F.softmax(heatmaps * beta, dim=-1)  # weighted probabilities
    heatmaps = heatmaps.view(B, K, H, W)

    # Create coordinate grids
    xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device)
    xs = xs.view(1, 1, 1, W)
    ys = ys.view(1, 1, H, 1)

    x = (heatmaps * xs).sum(dim=(2, 3))
    y = (heatmaps * ys).sum(dim=(2, 3))
    keypoints = torch.stack([x, y], dim=-1)
    return keypoints