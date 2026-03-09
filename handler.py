import cv2
import torch
import numpy as np
from utils import softargmax_2d, calc_angle, calc_signed_angle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_crop(image, center, crop_size):

    H, W = image.shape

    if H > 5500:
        crop_size = (crop_size[0] * 2, crop_size[1] * 2)    
    
    half = (crop_size[0] // 2, crop_size[1] // 2)

    x_center, y_center = center

    # Ensure coordinates stay within bounds
    x_center = np.clip(x_center, half[1], W - half[1])
    y_center = np.clip(y_center, half[0], H - half[0])

    # Bounding box coordinates
    x1 =int(max(0, x_center - half[1]))
    y1 = int(max(0, y_center - half[0]))
    x2 = int(min(W, x_center + half[1]))
    y2 = int(min(H, y_center + half[0]))
    
    cropped_image = image[y1:y2, x1:x2]
    
    return cropped_image, (x1, y1)

def process(image, ml_models):

    H, W = image.shape

    resized_image = cv2.resize(image, (256, 512))
    image_tensor = torch.from_numpy(resized_image).float().unsqueeze(0) / 255.0

    # ROI

    with torch.no_grad():
        roi_model = ml_models.models["roi"]
        inp = image_tensor.unsqueeze(0).to(device)
        outs = roi_model(inp)

    p_hmps = outs[-1]
    p_kps = softargmax_2d(p_hmps, beta=100.0)
    p_kps = p_kps.squeeze().cpu().numpy()

    p_kps[:, 0] *= 256 / 64
    p_kps[:, 1] *= 512 / 128

    # HIP
    h_cc = p_kps[0].copy()
    h_cc[0] *= W/256
    h_cc[1] *= H/512

    # KNEE
    k_cc = np.sum(p_kps[1:3], axis=0) / 2
    k_cc[0] *= W/256
    k_cc[1] *= H/512
    
    # ANKLE
    a_cc = p_kps[3].copy()
    a_cc[0] *= W/256
    a_cc[1] *= H/512

    h_cimg, h_shift = apply_crop(image, h_cc, crop_size=(384, 384))
    k_cimg, k_shift = apply_crop(image, k_cc, crop_size=(512, 512))
    a_cimg, a_shift = apply_crop(image, a_cc, crop_size=(256, 256))

    # HIP
    h_img = cv2.resize(h_cimg, (384, 384))
    h_img = torch.from_numpy(h_img).float().unsqueeze(0) / 255.0
    
    with torch.no_grad():
            hip_model = ml_models.models["hip"]
            h_inp = h_img.unsqueeze(0).to(device)
            h_out = hip_model(h_inp)
    
    hp_hmp = h_out[-1]
    
    h_img = h_img.squeeze().numpy()
    
    hp_kp = softargmax_2d(hp_hmp, beta=100.0)
    hp_kp = hp_kp.squeeze().cpu().numpy()
    hp_kp[0] /= 96 / h_cimg.shape[1]
    hp_kp[1] /= 96 / h_cimg.shape[0]
    hp_kp += h_shift
    
    # KNEE
    k_img = cv2.resize(k_cimg, (512, 512))
    k_img = torch.from_numpy(k_img).float().unsqueeze(0) / 255.0
    
    with torch.no_grad():
        knee_model = ml_models.models["knee"]
        k_inp = k_img.unsqueeze(0).to(device)
        k_outs = knee_model(k_inp)
    
    kp_hmps = k_outs[-1]
    
    k_img = k_img.squeeze().numpy()
    
    kp_kps = softargmax_2d(kp_hmps, beta=100.0)
    kp_kps = kp_kps.squeeze().cpu().numpy()
    kp_kps[:, 0] /= 128/ k_cimg.shape[1]
    kp_kps[:, 1] /= 128/ k_cimg.shape[0]
    kp_kps += k_shift
    
    # ANKLE
    a_img = cv2.resize(a_cimg, (256, 256))
    a_img = torch.from_numpy(a_img).float().unsqueeze(0) / 255.0
    
    with torch.no_grad():
        ankle_model = ml_models.models["ankle"]
        a_inp = a_img.unsqueeze(0).to(device)
        a_out = ankle_model(a_inp)
    
    ap_hmp = a_out[-1]
    
    a_img = a_img.squeeze().numpy()
    
    ap_kp = softargmax_2d(ap_hmp, beta=100.0)
    ap_kp = ap_kp.squeeze().cpu().numpy()
    ap_kp[0] /= 64 / a_cimg.shape[1]
    ap_kp[1] /= 64 / a_cimg.shape[0]
    ap_kp += a_shift
    
    keys = ["RM1", "RM2", "RSL1", "RSM1", "LTE", "MTE", "LTL", "MTM", "LTC", "MTC", "LTM", "MTL"]

    data = {key: val.tolist() for key, val in zip(keys, kp_kps)}

    data.update(
        {
             "R1": hp_kp.tolist(),
             "R4": ap_kp.tolist() 
        }
    )

    return data

    """
    HKA = calc_signed_angle(ap_kp, kp_kps[1], hp_kp, kp_kps[0])
    JLCA = calc_angle(kp_kps[2], kp_kps[3], kp_kps[8], kp_kps[9])
    MPTA = calc_angle(ap_kp, kp_kps[1], kp_kps[8], kp_kps[9])
    LDFA = calc_angle(hp_kp, kp_kps[0], kp_kps[2], kp_kps[3])

    return {
        "HKA": str(HKA),
        "JLCA": str(JLCA),
        "MPTA": str(MPTA),
        "LDFA": str(LDFA)
    }
    """