from pathlib import Path
import cv2
import torch
from torchvision.transforms import Compose, ToPILImage, ToTensor
from pia.ai.tasks.T2VRet.models.clip4clip.main import Clip4Clip, VisualModel
from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.model import PiaTorchModel

from devmacs_core.devmacs_core import DevMACSCore
from devmacs_core.utils.device import get_device

import os
from tqdm import tqdm

MODEL_PATH = Path("tests/large_files/clip4clip.pt")
model = DevMACSCore(
	model = MODEL_PATH,
	tile_size = "S",
)

video_path = "tests/large_files/falldown.mp4"
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

save_dir = "frame_vectors"
os.makedirs(save_dir, exist_ok=True)

for frame_idx in tqdm(range(total_frames), desc="Extracting Vectors", unit="frame"):
    ret, frame = cap.read()
    if not ret:
        break
    frame_vector = model.get_video_vector(frame)
    save_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.pt")
    torch.save(frame_vector, save_path)
cap.release()
print(f"\nAll frame vectors saved in '{save_dir}/'")