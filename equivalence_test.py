import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO

img = cv2.imread("bus.jpg")
img = cv2.resize(img, (640, 640))
img = img[:, :, ::-1].transpose(2, 0, 1)
x = np.ascontiguousarray(img, dtype=np.float32)[None] / 255.0

torch_model = YOLO("yolov8n.pt").model.float().eval()
with torch.no_grad():
    out_pt = torch_model(torch.from_numpy(x))[0].numpy()

session = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
out_onnx = session.run(None, {"images": x})[0]

diff = np.abs(out_pt - out_onnx)
print("Output shapes:", out_pt.shape, out_onnx.shape)
print(f"Max absolute difference:  {diff.max():.6f}")
print(f"Mean absolute difference: {diff.mean():.8f}")