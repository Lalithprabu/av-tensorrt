# hello_budget.py - a simple perception latency budget check
frame_rate_hz = 30
budget_ms = 1000 / frame_rate_hz

stages = {
    "camera capture + ISP": 8.0,
    "preprocessing": 3.0,
    "neural network inference": 8.3,
    "postprocessing (NMS)": 2.0,
    "hand-off to planning": 1.0,
}

total = sum(stages.values())
print(f"Frame budget at {frame_rate_hz} Hz: {budget_ms:.1f} ms")
for name, ms in stages.items():
    print(f"  {name:<28}{ms:>5.1f} ms")
print(f"Total used: {total:.1f} ms, margin: {budget_ms - total:.1f} ms")
print("BUDGET VIOLATED" if total > budget_ms else "Within budget")