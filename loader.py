import torch
from models import StackedHourglassCBAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h_model = StackedHourglassCBAM(num_keypoints=1, num_stacks=2, depth=4, channels=256, in_ch=1).to(device)
h_model.load_state_dict(torch.load('saved/hourglass_cbam[hip].pth', weights_only=True))
h_model.to(device)
h_model.eval()

k_model = StackedHourglassCBAM(num_keypoints=12, num_stacks=2, depth=4, channels=256, in_ch=1).to(device) # 12
k_model.load_state_dict(torch.load('saved/hourglass_cbam[knee].pth', weights_only=True))
k_model.to(device)
k_model.eval()

a_model = StackedHourglassCBAM(num_keypoints=1, num_stacks=2, depth=4, channels=256, in_ch=1).to(device)
a_model.load_state_dict(torch.load('saved/hourglass_cbam[ankle].pth', weights_only=True))
a_model.to(device)
a_model.eval()

roi_model = StackedHourglassCBAM(num_keypoints=4, num_stacks=2, depth=4, channels=256, in_ch=1).to(device)
roi_model.load_state_dict(torch.load('saved/hourglass_cbam_mse[roi].pth', weights_only=True))
roi_model.to(device)
roi_model.eval()
