import math
import torch
import numpy as np
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

def slope(P, Q, eps=1e-6):
    dx = Q[0] - P[0]
    dy = Q[1] - P[1]
    if abs(dx) < eps:
        return float('inf')  # vertical line
    return dy / dx

def calc_tan_angle(m1, m2, eps=1e-6):
    # vertical vs vertical
    if math.isinf(m1) and math.isinf(m2):
        return 0.0

    # one vertical, one not
    if math.isinf(m1) and not math.isinf(m2):
        if abs(m2) < eps:  # vertical vs horizontal
            return 90.0
        return math.degrees(math.atan(abs(1.0 / m2)))

    if not math.isinf(m1) and math.isinf(m2):
        if abs(m1) < eps:
            return 90.0
        return math.degrees(math.atan(abs(1.0 / m1)))

    # general case
    denom = 1 + m1 * m2
    if abs(denom) < eps:
        return 90.0  # perpendicular (or very close)

    tan_theta = abs((m1 - m2) / denom)
    angle_rad = math.atan(tan_theta)
    return math.degrees(angle_rad)

def calc_angle(A, B, C, D):
    m_ab = slope(A, B)
    m_cd = slope(C, D)
    return calc_tan_angle(m_ab, m_cd)

def calc_signed_angle(A, B, C, D):
    """
    Calculates the HKA angle given four points:
    A: Femur Head Center
    B: Knee Center (Femur side)
    C: Knee Center (Tibia side - usually same as B)
    D: Ankle Center
    """
    # Create vectors for the Femoral Mechanical Axis (AB) and Tibial Mechanical Axis (CD)
    vector_AB = np.array(B) - np.array(A)
    vector_CD = np.array(D) - np.array(C)

    # Calculate the angle of each vector relative to the horizontal axis
    angle_AB = np.arctan2(vector_AB[1], vector_AB[0])
    angle_CD = np.arctan2(vector_CD[1], vector_CD[0])

    # The difference between them is the intersection angle
    angle_rad = angle_AB - angle_CD
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    # Normalize the result to clinical standards 
    # (where 180 is straight, <180 is Varus, >180 is Valgus)
    hka_angle = 180 - abs(angle_deg)

    return hka_angle