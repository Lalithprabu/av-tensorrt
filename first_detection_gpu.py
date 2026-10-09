from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = "https://ultralytics.com/images/bus.jpg"

for run in range(1, 6):
    results = model(image, device=0, verbose=False)
    speed = results[0].speed
    print(f"Run {run}: preprocess {speed['preprocess']:.1f} ms, "
          f"inference {speed['inference']:.1f} ms, "
          f"postprocess {speed['postprocess']:.1f} ms")

results[0].save("bus_detected_gpu.jpg")