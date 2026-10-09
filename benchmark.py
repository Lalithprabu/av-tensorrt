import time
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
image = "bus.jpg"

WARMUP_RUNS = 10
TIMED_RUNS = 300
INFERENCE_BUDGET_MS = 15.0

print("Warming up...")
for _ in range(WARMUP_RUNS):
    model(image, device=0, verbose=False)

inference_ms = []
total_ms = []
print(f"Measuring {TIMED_RUNS} runs...")
for _ in range(TIMED_RUNS):
    start = time.perf_counter()
    results = model(image, device=0, verbose=False)
    end = time.perf_counter()
    inference_ms.append(results[0].speed["inference"])
    total_ms.append((end - start) * 1000)

def report(name, values):
    values = np.array(values)
    print(f"{name:<12} mean {values.mean():6.2f}   p99 {np.percentile(values, 99):6.2f}   max {values.max():6.2f} ms")

report("Inference", inference_ms)
report("End-to-end", total_ms)

misses = sum(1 for v in inference_ms if v > INFERENCE_BUDGET_MS)
print(f"Inference budget misses (> {INFERENCE_BUDGET_MS} ms): {misses} of {TIMED_RUNS}")