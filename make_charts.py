# make_charts.py
# Builds every image in the README from results.json.
# Workflow after each phase:
#   1. Put your new measured numbers into results.json
#   2. Run:  python make_charts.py
#   3. Commit and push. The README pictures update automatically.
#
# Each chart is made twice (light and dark) so it matches the
# viewer's GitHub theme.

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
DATA = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))

FONT = "Segoe UI, Inter, -apple-system, Helvetica, Arial, sans-serif"
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "svg.fonttype": "path",  # text becomes shapes, so it looks identical everywhere
    "axes.linewidth": 1,
})

# Colours for each theme. Background values match GitHub's own page colours.
THEMES = {
    "light": {
        "bg": "#ffffff", "ink": "#1f2328", "ink2": "#59636e", "muted": "#818b98",
        "grid": "#e8eaed", "axis": "#d0d7de", "track": "#eef1f4",
        "blue": "#2a78d6", "good": "#0ca30c", "critical": "#d03b3b",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    },
    "dark": {
        "bg": "#0d1117", "ink": "#e6edf3", "ink2": "#9198a1", "muted": "#6e7681",
        "grid": "#21262d", "axis": "#30363d", "track": "#161b22",
        "blue": "#3987e5", "good": "#0ca30c", "critical": "#d03b3b",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    },
}


def text_on(fill_hex):
    """White or dark text, whichever is readable on a coloured fill."""
    r, g, b = (int(fill_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.55 else "#ffffff"


def new_figure(t, width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    return fig, ax


def titles(fig, t, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.18 / h, title, color=t["ink"], fontsize=15, weight="bold", va="top")
    fig.text(0.012, 1 - 0.52 / h, subtitle, color=t["ink2"], fontsize=10.5, va="top")


def reference_line(ax, t, x, label, strong=False, y_text=None):
    ax.axvline(x, color=t["ink"] if strong else t["muted"], lw=1.4 if strong else 1, zorder=1)
    ax.text(x + 0.6, y_text, label, color=t["ink"] if strong else t["ink2"],
            fontsize=9.5, va="bottom", weight="bold" if strong else "normal")


def rounded_bar(ax, x0, y, width, height, color, zorder=3):
    ax.add_patch(FancyBboxPatch((x0, y - height / 2), width, height,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                mutation_aspect=1 / 6, fc=color, ec="none", zorder=zorder))


def save(fig, name, theme):
    path = ASSETS / f"{name}_{theme}.svg"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------- chart 1
def latency_chart(theme):
    t = THEMES[theme]
    budget = DATA["frame_budget_ms"]
    alloc = DATA["inference_allocation_ms"]
    configs = DATA["latency_configs"]
    fig, ax = new_figure(t, 10, 3.9)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.70, bottom=0.12)
    titles(fig, t, "Inference latency per frame",
           f"{DATA['model']}  ·  mean inference time  ·  vertical lines are the automotive budget")

    xmax = 150
    rows = list(range(len(configs)))[::-1]
    for y, c in zip(rows, configs):
        v = c["inference_mean_ms"]
        if v is None:
            rounded_bar(ax, 0, y, xmax, 0.42, t["track"], zorder=2)
            ax.text(1.5, y, f"pending  ·  {c['note']}", color=t["muted"],
                    fontsize=10, va="center", style="italic", zorder=4)
            continue
        over = v > budget
        colour = t["critical"] if over else t["good"]
        rounded_bar(ax, 0, y, v, 0.42, colour)
        value = f"{v:.1f} ms" if v >= 100 else f"{v:.2f} ms"
        if over:  # long bar: write inside it, right-aligned
            ax.text(v - 2, y, f"{value}   ·   OVER BUDGET  ({v / budget:.1f}× the frame)",
                    color="#ffffff", fontsize=11, weight="bold", va="center", ha="right", zorder=5)
        else:
            ax.text(v + 1.5, y, value, color=t["ink"], fontsize=11.5, weight="bold", va="center")
            ax.text(v + 15.5, y, "WITHIN BUDGET", color=t["good"], fontsize=9.5,
                    weight="bold", va="center")

    ax.set_yticks(rows)
    ax.set_yticklabels([f"{c['label']}\n{c['note']}" for c in configs],
                       color=t["ink"], fontsize=10.5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(configs) + 0.25)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax.set_xlabel("milliseconds", color=t["muted"], fontsize=9.5)
    reference_line(ax, t, alloc, "inference slot 15 ms", y_text=len(configs) - 0.6)
    reference_line(ax, t, budget, "frame budget 33.3 ms", strong=True, y_text=len(configs) - 0.12)
    save(fig, "latency", theme)


# ---------------------------------------------------------------- chart 2
def timing_spread_chart(theme):
    t = THEMES[theme]
    s = DATA["timing_spread_fp32_gpu"]
    fig, ax = new_figure(t, 10, 3.5)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.68, bottom=0.15)
    titles(fig, t, "Typical vs worst-case frame time  (GPU, FP32)",
           f"{s['runs']} timed runs after 10 warm-up runs  ·  safety arguments use the worst case, not the mean")

    rows = [("Inference", s["inference"], 1), ("End-to-end", s["end_to_end"], 0)]
    shapes = {"mean": "o", "p99": "D", "max": "s"}
    for label, d, y in rows:
        ax.plot([d["mean"], d["max"]], [y, y], color=t["blue"], lw=2.5,
                solid_capstyle="round", zorder=3)
        for key, marker in shapes.items():
            ax.plot(d[key], y, marker=marker, ms=10 if key == "mean" else 8.5,
                    color=t["blue"], mec=t["bg"], mew=2, zorder=4)
            below = key == "p99"
            ax.text(d[key], y - 0.2 if below else y + 0.2, f"{key} {d[key]:.1f}",
                    color=t["ink"], fontsize=9.5, ha="center", va="top" if below else "bottom")

    for key, marker in shapes.items():
        ax.plot([], [], marker=marker, ls="none", color=t["ink2"], ms=8, label=key)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False,
                    fontsize=10, handletextpad=0.3, columnspacing=1.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Inference", "End-to-end"], color=t["ink"], fontsize=11)
    ax.set_xlim(0, 36)
    ax.set_ylim(-0.55, 1.65)
    ax.set_xlabel("milliseconds", color=t["muted"], fontsize=9.5)
    reference_line(ax, t, DATA["inference_allocation_ms"], "inference slot 15 ms", y_text=1.42)
    reference_line(ax, t, DATA["frame_budget_ms"], "frame\n33.3 ms", strong=True, y_text=1.2)
    save(fig, "timing_spread", theme)


# ---------------------------------------------------------------- chart 3
def budget_chart(theme):
    t = THEMES[theme]
    budget = DATA["frame_budget_ms"]
    stages = DATA["budget_model"]
    fig, ax = new_figure(t, 10, 2.2)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.62, bottom=0.04)
    titles(fig, t, "Where the 33.3 ms frame goes",
           "Inference = measured worst case of 300 runs.  Other stages = allocations, not yet measured.")
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    rounded_bar(ax, 0, 0, budget, 0.62, t["track"], zorder=1)
    x = 0
    for colour, st in zip(t["series"], stages):
        w = st["ms"]
        ax.barh(0, w, left=x, height=0.62, color=colour, edgecolor=t["bg"], lw=2, zorder=3)
        if w >= 2.5:
            ax.text(x + w / 2, 0, f"{w:.1f}", color=text_on(colour), fontsize=10.5,
                    ha="center", va="center", weight="bold", zorder=4)
        x += w
    margin = budget - x
    ax.text(x + margin / 2, 0, f"margin {margin:.1f} ms  ({margin / budget:.0%})",
            color=t["ink2"], fontsize=10.5, ha="center", va="center", zorder=4)

    # legend row underneath: swatch + name + value + measured/allocated
    lx = 0
    for colour, st in zip(t["series"], stages):
        ax.add_patch(FancyBboxPatch((lx, -0.92), 0.55, 0.26, boxstyle="round,pad=0,rounding_size=0.08",
                                    mutation_aspect=1 / 6, fc=colour, ec="none"))
        tag = "measured" if st["measured"] else "allocated"
        ax.text(lx + 0.8, -0.79, f"{st['stage']}  {st['ms']:.1f} ms", color=t["ink"],
                fontsize=9.5, va="center")
        ax.text(lx + 0.8, -1.13, tag, color=t["good"] if st["measured"] else t["muted"],
                fontsize=8.5, va="center", weight="bold" if st["measured"] else "normal")
        lx += budget / len(stages)
    ax.set_xlim(0, budget)
    ax.set_ylim(-1.35, 0.45)
    save(fig, "budget", theme)


# ---------------------------------------------------------------- diagram
def scales_diagram(theme):
    t = THEMES[theme]
    panel, gap, top = 270, 50, 128
    width = 3 * panel + 2 * gap + 80
    height = top + panel + 120
    grids = [(80, 8, "6,400", "small objects"), (40, 16, "1,600", "medium objects"),
             (20, 32, "400", "large, near objects")]
    ped = (292, 258, 16, 40)  # a distant pedestrian: x, y, w, h in the 640 px input
    s = panel / 640

    def cells_on(stride):
        px, py, pw, ph = ped
        return ((int((px + pw) / stride) - int(px / stride) + 1)
                * (int((py + ph) / stride) - int(py / stride) + 1))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="{FONT}">',
           f'<rect width="100%" height="100%" fill="{t["bg"]}"/>',
           f'<text x="40" y="44" fill="{t["ink"]}" font-size="21" font-weight="700">'
           'YOLOv8 looks at every frame on three grids</text>',
           f'<text x="40" y="74" fill="{t["ink2"]}" font-size="14.5">The same distant pedestrian '
           f'(about 16 × 40 px) lands on {cells_on(8)} cells of the finest grid but only {cells_on(32)} of the coarsest.</text>',
           f'<text x="40" y="96" fill="{t["ink2"]}" font-size="14.5">Small, distant objects depend on the fine grid, '
           'so that branch is the first suspect when INT8 costs accuracy.</text>']
    for i, (n, stride, cands, role) in enumerate(grids):
        x0 = 40 + i * (panel + gap)
        y0 = top
        cell = panel / n
        out.append(f'<rect x="{x0}" y="{y0}" width="{panel}" height="{panel}" rx="6" '
                   f'fill="{t["track"]}" stroke="{t["axis"]}"/>')
        lines = []
        for k in range(1, n):
            p = k * cell
            lines.append(f'M{x0 + p:.2f} {y0}V{y0 + panel}M{x0} {y0 + p:.2f}H{x0 + panel}')
        out.append(f'<path d="{"".join(lines)}" stroke="{t["axis"]}" stroke-width="0.6" fill="none"/>')
        # cells the pedestrian touches
        px, py, pw, ph = ped
        c0, c1 = int(px / stride), int((px + pw) / stride)
        r0, r1 = int(py / stride), int((py + ph) / stride)
        out.append(f'<rect x="{x0 + c0 * cell:.2f}" y="{y0 + r0 * cell:.2f}" '
                   f'width="{(c1 - c0 + 1) * cell:.2f}" height="{(r1 - r0 + 1) * cell:.2f}" '
                   f'fill="{t["series"][0]}" fill-opacity="0.28" stroke="{t["series"][0]}" stroke-width="1.2"/>')
        out.append(f'<rect x="{x0 + px * s:.2f}" y="{y0 + py * s:.2f}" width="{pw * s:.2f}" '
                   f'height="{ph * s:.2f}" rx="2" fill="{t["series"][1]}"/>')
        # zoom inset: 64 x 64 input pixels around the pedestrian, magnified
        zin, zsize = 64, 118
        k = zsize / zin
        zx0, zy0 = px + pw / 2 - zin / 2, py + ph / 2 - zin / 2  # region origin, input px
        ix, iy = x0 + panel - zsize - 12, y0 + panel - zsize - 12
        cid = f"clip{theme}{i}"
        out.append(f'<clipPath id="{cid}"><rect x="{ix}" y="{iy}" width="{zsize}" height="{zsize}" rx="6"/></clipPath>')
        out.append(f'<line x1="{x0 + (px + pw) * s:.1f}" y1="{y0 + (py + ph) * s:.1f}" x2="{ix}" y2="{iy}" '
                   f'stroke="{t["ink2"]}" stroke-width="1"/>')
        out.append(f'<g clip-path="url(#{cid})"><rect x="{ix}" y="{iy}" width="{zsize}" height="{zsize}" fill="{t["bg"]}"/>')
        zl = []
        first = int(zx0 // stride) * stride
        for g in range(int(first), int(zx0 + zin) + stride, stride):
            gx = ix + (g - zx0) * k
            zl.append(f"M{gx:.1f} {iy}V{iy + zsize}")
        first = int(zy0 // stride) * stride
        for g in range(int(first), int(zy0 + zin) + stride, stride):
            gy = iy + (g - zy0) * k
            zl.append(f"M{ix} {gy:.1f}H{ix + zsize}")
        out.append(f'<path d="{"".join(zl)}" stroke="{t["muted"]}" stroke-width="1" fill="none"/>')
        out.append(f'<rect x="{ix + (c0 * stride - zx0) * k:.1f}" y="{iy + (r0 * stride - zy0) * k:.1f}" '
                   f'width="{(c1 - c0 + 1) * stride * k:.1f}" height="{(r1 - r0 + 1) * stride * k:.1f}" '
                   f'fill="{t["series"][0]}" fill-opacity="0.3" stroke="{t["series"][0]}" stroke-width="1.5"/>')
        out.append(f'<rect x="{ix + (px - zx0) * k:.1f}" y="{iy + (py - zy0) * k:.1f}" width="{pw * k:.1f}" '
                   f'height="{ph * k:.1f}" rx="3" fill="{t["series"][1]}"/></g>')
        cells = (c1 - c0 + 1) * (r1 - r0 + 1)
        out.append(f'<rect x="{ix}" y="{iy}" width="{zsize}" height="{zsize}" rx="6" fill="none" '
                   f'stroke="{t["ink2"]}" stroke-width="1.2"/>')
        out.append(f'<text x="{ix + zsize / 2}" y="{iy - 8}" fill="{t["ink"]}" font-size="12" '
                   f'font-weight="700" text-anchor="middle">zoom: {cells} cell{"s" if cells > 1 else ""}</text>')
        if i == 0:
            out.append(f'<rect x="{x0 - 5}" y="{y0 - 5}" width="{panel + 10}" height="{panel + 10}" '
                       f'rx="9" fill="none" stroke="{t["blue"]}" stroke-width="2.5"/>')
            out.append(f'<rect x="{x0 + 12}" y="{y0 + 12}" width="168" height="26" rx="13" fill="{t["blue"]}"/>'
                       f'<text x="{x0 + 96}" y="{y0 + 30}" fill="#ffffff" font-size="12.5" '
                       'font-weight="700" text-anchor="middle">Watch this under INT8</text>')
        ty = y0 + panel + 34
        out.append(f'<text x="{x0}" y="{ty}" fill="{t["ink"]}" font-size="17" font-weight="700">'
                   f'{n} × {n} grid</text>')
        out.append(f'<text x="{x0}" y="{ty + 24}" fill="{t["ink2"]}" font-size="13.5">'
                   f'{cands} candidates  ·  each cell {stride} × {stride} px</text>')
        out.append(f'<text x="{x0}" y="{ty + 46}" fill="{t["ink2"]}" font-size="13.5">Best at {role}</text>')
    out.append('</svg>')
    path = ASSETS / f"detection_scales_{theme}.svg"
    path.write_text("\n".join(out), encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


# ---------------------------------------------------------------- banner
def banner():
    done, total = DATA["phase_done"], DATA["phase_total"]
    budget = DATA["frame_budget_ms"]
    gpu = DATA["timing_spread_fp32_gpu"]
    inf, e2e = gpu["inference"]["mean"], gpu["end_to_end"]["mean"]
    W, H = 1280, 340
    meter_x, meter_w = 760, 460
    stages = [("PyTorch FP32", 2), ("ONNX", 3), ("TensorRT FP16", 4), ("INT8 + mixed", 5)]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}">',
           '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
           '<stop offset="0" stop-color="#0a0f1a"/><stop offset="1" stop-color="#12284a"/></linearGradient>'
           '<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">'
           '<path d="M32 0H0V32" fill="none" stroke="#ffffff" stroke-opacity="0.05"/></pattern></defs>',
           f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>',
           f'<rect width="{W}" height="{H}" rx="18" fill="url(#grid)"/>',
           '<text x="56" y="70" fill="#7fb2f0" font-size="15" font-weight="700" letter-spacing="3">'
           'EMBEDDED AI  ·  AUTOMOTIVE PERCEPTION</text>',
           '<text x="54" y="138" fill="#ffffff" font-size="62" font-weight="800">AV TensorRT</text>',
           '<text x="56" y="180" fill="#c9d4e3" font-size="21">Optimising YOLO for a 33 ms frame budget,</text>',
           '<text x="56" y="208" fill="#c9d4e3" font-size="21">without losing the distant pedestrian.</text>']
    # pipeline chips
    cx = 56
    for name, phase in stages:
        state = "done" if phase <= done else ("next" if phase == done + 1 else "planned")
        w = 34 + 9.4 * len(name) + 2.2 * sum(ch.isupper() for ch in name) + (22 if state == "done" else 0)
        fill = {"done": "#1b6b3a", "next": "#5a4300", "planned": "#1c2a40"}[state]
        stroke = {"done": "#35c46a", "next": "#f0b429", "planned": "#3a4d6b"}[state]
        out.append(f'<rect x="{cx}" y="256" width="{w:.0f}" height="38" rx="19" fill="{fill}" stroke="{stroke}"/>')
        tx = cx + 16
        if state == "done":
            out.append(f'<path d="M{tx} 275l5 5 9-10" stroke="#35c46a" stroke-width="2.6" fill="none" '
                       'stroke-linecap="round" stroke-linejoin="round"/>')
            tx += 22
        out.append(f'<text x="{tx}" y="281" fill="#ffffff" font-size="15" font-weight="600">{name}</text>')
        cx += w + 12
        if phase != stages[-1][1]:
            out.append(f'<path d="M{cx - 9} 275h6" stroke="#5f7393" stroke-width="2"/>')
    # budget meter
    sc = meter_w / budget
    out += [f'<text x="{meter_x}" y="92" fill="#c9d4e3" font-size="15">Frame budget  ·  30 fps</text>',
            f'<text x="{meter_x + meter_w}" y="92" fill="#ffffff" font-size="15" font-weight="700" '
            f'text-anchor="end">{budget} ms</text>',
            f'<rect x="{meter_x}" y="106" width="{meter_w}" height="22" rx="11" fill="#1c2a40"/>',
            f'<rect x="{meter_x}" y="106" width="{e2e * sc:.1f}" height="22" rx="11" fill="#2a78d6" fill-opacity="0.45"/>',
            f'<rect x="{meter_x}" y="106" width="{inf * sc:.1f}" height="22" rx="11" fill="#5aa2f5"/>',
            f'<text x="{meter_x}" y="190" fill="#ffffff" font-size="54" font-weight="800">{inf:.2f}'
            '<tspan font-size="24" font-weight="600" fill="#c9d4e3"> ms</tspan></text>',
            f'<text x="{meter_x}" y="218" fill="#9fb0c8" font-size="14.5">mean GPU inference, FP32  ·  '
            f'{e2e:.1f} ms end-to-end</text>']
    # phase dots
    out.append(f'<text x="{meter_x}" y="281" fill="#9fb0c8" font-size="14.5">Phase {done} of {total}</text>')
    for k in range(total):
        fill = "#35c46a" if k < done else ("#f0b429" if k == done else "#2b3b55")
        out.append(f'<circle cx="{meter_x + 130 + k * 26}" cy="276" r="7" fill="{fill}"/>')
    out.append('</svg>')
    path = ASSETS / "banner.svg"
    path.write_text("\n".join(out), encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    for theme in THEMES:
        latency_chart(theme)
        timing_spread_chart(theme)
        budget_chart(theme)
        scales_diagram(theme)
    banner()
    print("Done. Charts are in the assets folder.")
