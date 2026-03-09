import cv2
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from handler import process
from models import StackedHourglassCBAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelContainer:
    def __init__(self):
        self.device = device
        self.models = {}

    def load_hourglass(self, name, num_kpts, path):
        model = StackedHourglassCBAM(
            num_keypoints=num_kpts, num_stacks=2, depth=4, channels=256, in_ch=1
        ).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()
        self.models[name] = model

# Initialize the container
ml_models = ModelContainer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models on startup
    print(f"Loading models on {ml_models.device}...")
    ml_models.load_hourglass("hip", 1, 'saved/hourglass_cbam[hip].pth')
    ml_models.load_hourglass("knee", 12, 'saved/hourglass_cbam[knee].pth')
    ml_models.load_hourglass("ankle", 1, 'saved/hourglass_cbam[ankle].pth')
    ml_models.load_hourglass("roi", 4, 'saved/hourglass_cbam_mse[roi].pth')
    yield
    # Clean up (clear GPU cache if needed)
    ml_models.models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)



# Define allowed extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "bmp"}

@app.post("/upload-image/")
async def upload_picture(file: UploadFile = File(...)):
    # 1. Validate File Extension
    extension = file.filename.split(".")[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only JPG and BMP are allowed.")
    
    # 2. Read file bytes
    contents = await file.read()
    # 3. Convert bytes to NumPy array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    return JSONResponse(content=process(image, ml_models), status_code=200)

