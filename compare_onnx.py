from ultralytics import YOLO

for name in ["yolov8n.pt", "yolov8n.onnx"]:
    model = YOLO(name)
    results = model("bus.jpg", device="cpu", verbose=False)
    boxes = results[0].boxes
    print(f"\n{name}: {len(boxes)} objects")
    for cls, conf in zip(boxes.cls, boxes.conf):
        print(f"  {model.names[int(cls)]:<10} {float(conf):.3f}")