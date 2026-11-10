"""Generate docs/architectures.svg, the animated figure of the README.

Usage:
    python tools/build_architectures_svg.py [OUTPUT]

Needs only the standard library. A binned MALDI-TOF spectrum is routed in turn
through the four backbones (MLP, 1-D CNN, 1-D ResNet, 1-D ViT): the active card
lights up, its schematic animates, then the R / S probability bars update. The
spectrum (fixed seed, so the output is reproducible) and the probabilities are
illustrative, they are not computed by MaldiDeepKit.

The animation is pure SMIL: GitHub shows README images through <img>, where
JavaScript and external files do not run. For the same reason colours are
mid-tones readable on both themes: inside <img>, prefers-color-scheme follows
the operating system, not the GitHub theme.
"""

import math
import random
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs" / "architectures.svg"

LOOP = 12.0  # seconds
PH = 3.0  # seconds per backbone
ACC = "#0072B2"  # active accent (Okabe-Ito blue)
GREY = "#7d8590"  # strokes / shapes, readable on white and #0d1117
TXT = "#6e7781"  # text, ~4.5:1 on white and ~4.2:1 on #0d1117
R_C = "#D55E00"  # Resistant (vermillion)
S_C = "#009E73"  # Susceptible (bluish green)
EASE = ".45 0 .25 1"

MODELS = [
    ("MLP", "MaldiMLPClassifier", 0.82),
    ("1-D CNN", "MaldiCNNClassifier", 0.88),
    ("1-D ResNet", "MaldiResNetClassifier", 0.91),
    ("1-D ViT", "MaldiTransformerClassifier", 0.86),
]

rng = random.Random(20240501)


def n(x):
    """Format a number with at most two decimals."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def kt(t):
    """Convert a time in seconds to a SMIL keyTimes value."""
    s = f"{t / LOOP:.4f}".rstrip("0").rstrip(".")
    return s or "0"


def anim(attr, kf, spline=False, extra=""):
    """Return an animation over the 12 s loop from (t, value) keyframes."""
    kf = sorted(kf, key=lambda p: p[0])
    if kf[0][0] > 0:
        kf.insert(0, (0.0, kf[0][1]))
    if kf[-1][0] < LOOP:
        kf.append((LOOP, kf[-1][1]))
    clean = []
    for t, v in kf:
        if clean and abs(clean[-1][0] - t) < 1e-9:
            clean[-1] = (t, v)
        else:
            clean.append((t, v))
    times = ";".join(kt(t) for t, _ in clean)
    vals = ";".join(v if isinstance(v, str) else n(v) for _, v in clean)
    s = (
        f'<animate attributeName="{attr}" dur="{n(LOOP)}s" repeatCount="indefinite" '
        f'keyTimes="{times}" values="{vals}"'
    )
    if spline:
        s += (
            ' calcMode="spline" keySplines="'
            + ";".join([EASE] * (len(clean) - 1))
            + '"'
        )
    return s + extra + "/>"


def pulse(S, t_on, v_on, base=0.0, ramp=0.12, t_off=2.85, off_ramp=0.15):
    """Return keyframes ramping base to v_on at S+t_on, back at S+t_off."""
    return [
        (S + t_on, base),
        (S + t_on + ramp, v_on),
        (S + t_off, v_on),
        (S + t_off + off_ramp, base),
    ]


def colour_kf(S, t_on=0.6, ramp=0.25, t_off=2.85, off_ramp=0.15, base=TXT, on=ACC):
    """Return keyframes switching a colour on at S+t_on and off at S+t_off."""
    return [
        (S + t_on, base),
        (S + t_on + ramp, on),
        (S + t_off, on),
        (S + t_off + off_ramp, base),
    ]


def motion(path, S, t0, t1):
    """Return an animateMotion along path during [S+t0, S+t1], parked otherwise."""
    a, b = S + t0, S + t1
    times, pts = [0.0], ["0"]
    if a > 0:
        times.append(a)
        pts.append("0")
    times += [b, LOOP]
    pts += ["1", "1"]
    return (
        f'<animateMotion dur="{n(LOOP)}s" repeatCount="indefinite" calcMode="linear" '
        f'path="{path}" keyTimes="{";".join(kt(t) for t in times)}" keyPoints="{";".join(pts)}"/>'
    )


PEAKS = [
    (0.045, 0.42),
    (0.09, 0.22),
    (0.155, 0.95),
    (0.21, 0.3),
    (0.27, 0.62),
    (0.33, 0.16),
    (0.405, 1.0),
    (0.47, 0.26),
    (0.545, 0.5),
    (0.62, 0.14),
    (0.7, 0.34),
    (0.79, 0.12),
    (0.87, 0.2),
    (0.94, 0.09),
]

NB = 64
SX0, SX1, SBASE, SH = 16.0, 176.0, 186.0, 70.0
pitch = (SX1 - SX0) / NB
bins = []
for i in range(NB):
    c = (i + 0.5) / NB
    v = 0.05 + 0.1 * math.exp(-4.0 * c) + rng.random() * 0.035
    for pc, h in PEAKS:
        d = abs(pc - c) * NB  # distance in bins
        v += h * math.exp(-0.5 * (d / 0.55) ** 2)
    bins.append(v)
top = max(bins)
bins = [b / top for b in bins]


def spec_bars(fill):
    """Return the binned input spectrum as bars."""
    out = []
    for i, b in enumerate(bins):
        h = max(1.2, b * SH)
        out.append(
            f'<rect x="{n(SX0 + i * pitch + 0.35)}" y="{n(SBASE - h)}" width="{n(pitch - 0.7)}" height="{n(h)}"/>'
        )
    return f'<g fill="{fill}">' + "".join(out) + "</g>"


def mini_spec(x0, x1, ybase, h, seed):
    """Return a compact spectrum trace with the peak list of the input panel."""
    r = random.Random(seed)
    xs = [x0 + i * 0.5 for i in range(int((x1 - x0) / 0.5) + 1)]
    pts = []
    for x in xs:
        u = (x - x0) / (x1 - x0)
        v = 0.04 + 0.08 * math.exp(-4.0 * u) + r.random() * 0.03
        for pc, ph in PEAKS:
            v += ph * math.exp(-0.5 * (((u - pc) * (x1 - x0)) / 0.9) ** 2)
        pts.append((x, ybase - min(v, 1.0) * h))
    # drop points on near-straight runs to keep the path small
    keep = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:], strict=False):
        if abs((a[1] + c[1]) / 2 - b[1]) > 0.12:
            keep.append(b)
    keep.append(pts[-1])
    return "M" + " L".join(f"{n(x)},{n(y)}" for x, y in keep)


CX, CW, CH = 228, 344, 50
CY = [40, 98, 156, 214]
HUB_L = (182, 152)
HUB_R = (612, 152)


def lpath(yc):
    """Return the connector from the input hub to a card."""
    x0, y0 = HUB_L
    return f"M{x0} {y0}C205 {y0} 205 {yc} {CX} {yc}"


def rpath(yc):
    """Return the connector from a card to the output hub."""
    x1, y1 = HUB_R
    return f"M{CX + CW} {yc}C592 {yc} 592 {y1} {x1} {y1}"


def mlp(S):
    """Return the MLP schematic of the phase starting at S."""
    layers = [4, 5, 3, 2]
    xs = [24, 62, 100, 138]
    pos = [
        [(xs[j], 25 + (i - (m - 1) / 2) * 7.5) for i in range(m)]
        for j, m in enumerate(layers)
    ]
    out = []
    edges = []
    for j in range(3):
        segs = "".join(
            f"M{n(a[0])} {n(a[1])}L{n(b[0])} {n(b[1])}"
            for a in pos[j]
            for b in pos[j + 1]
        )
        edges.append(segs)
    out.append(
        f'<g stroke="{GREY}" stroke-width=".7" stroke-opacity=".5">'
        + "".join(f'<path d="{e}"/>' for e in edges)
        + "</g>"
    )
    out.append(
        f'<g fill="{GREY}">'
        + "".join(
            f'<circle cx="{n(x)}" cy="{n(y)}" r="2.6"/>' for L in pos for x, y in L
        )
        + "</g>"
    )
    for j, e in enumerate(edges):
        out.append(
            f'<path d="{e}" stroke="{ACC}" stroke-width="1" opacity="0">'
            + anim("opacity", pulse(S, 1.0 + 0.3 * j, 0.75))
            + "</path>"
        )
    for j, L in enumerate(pos):
        out.append(
            f'<g fill="{ACC}" opacity="0">'
            + "".join(f'<circle cx="{n(x)}" cy="{n(y)}" r="3"/>' for x, y in L)
            + anim("opacity", pulse(S, 0.85 + 0.3 * j, 1))
            + "</g>"
        )
    return "".join(out)


def arrow(x0, x1, y):
    """Return a small horizontal arrow."""
    return (
        f'<path d="M{x0} {y}H{x1}M{x1 - 3} {y - 3}L{x1} {y}L{x1 - 3} {y + 3}" stroke="{GREY}" '
        'stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def cnn(S):
    """Return the 1-D CNN schematic of the phase starting at S."""
    out = [
        f'<path d="{mini_spec(12, 92, 40, 28, 3)}" stroke="{GREY}" stroke-width="1.1" stroke-linejoin="round"/>'
    ]
    out.append(arrow(98, 108, 25))
    # kernel window: grey at rest, accent while sliding
    out.append(
        f'<rect x="12" y="9" width="14" height="33" rx="2.5" stroke="{GREY}" stroke-width="1.3" '
        f'fill="{ACC}" fill-opacity="0" stroke-dasharray="3 2">'
        + anim("stroke", colour_kf(S, 0.6, 0.25, base=GREY))
        + anim("fill-opacity", pulse(S, 0.6, 0.16, ramp=0.25))
        + f'<animateTransform attributeName="transform" type="translate" dur="{n(LOOP)}s" '
        f'repeatCount="indefinite" keyTimes="0;{kt(S + 0.85)};{kt(S + 2.05)};{kt(S + 2.85)};'
        f'{kt(S + 3.0)};1" values="0 0;0 0;66 0;66 0;0 0;0 0"/></rect>'
    )
    cells_base, cells_acc = [], []
    for yy in (13, 22, 31):
        cells_base.append(
            f'<rect x="114" y="{yy}" width="42" height="6" rx="1" stroke="{GREY}" stroke-width=".8"/>'
        )
        for c in range(7):
            a = 0.25 + 0.75 * rng.random()
            cells_acc.append(
                f'<rect x="{n(114 + c * 6)}" y="{yy}" width="6" height="6" fill="{ACC}" fill-opacity="{n(a)}"/>'
            )
            cells_base.append(
                f'<rect x="{n(114 + c * 6)}" y="{yy}" width="6" height="6" fill="{GREY}" fill-opacity="{n(a * 0.35)}"/>'
            )
    out.append("<g>" + "".join(cells_base) + "</g>")
    out.append(
        '<g clip-path="url(#cf)">'
        + anim("opacity", [(S + 2.85, 1), (S + 3.0, 0), (S + 3.05, 0), (S + 3.1, 1)])
        + "".join(cells_acc)
        + "</g>"
    )
    return "".join(out)


def resnet(S):
    """Return the 1-D ResNet schematic of the phase starting at S."""
    out = []
    arcs = ["M16 33C16 13 74 13 74 28.5", "M80 33C80 13 140 13 140 28.5"]
    out.append(f'<path d="M12 33H158" stroke="{GREY}" stroke-width="1.1"/>')
    for a in arcs:
        out.append(
            f'<path d="{a}" stroke="{GREY}" stroke-width="1.2" stroke-opacity=".8"/>'
        )
    blocks = [22, 44, 88, 110]
    adds = [74, 140]
    for x in blocks:
        out.append(
            f'<rect x="{x}" y="27" width="18" height="12" rx="2" stroke="{GREY}" stroke-width="1.1" '
            f'fill="{GREY}" fill-opacity=".18"/>'
        )
    for x in adds:
        out.append(
            f'<circle cx="{x}" cy="33" r="4.5" stroke="{GREY}" stroke-width="1.1" fill="{GREY}" fill-opacity=".12"/>'
            f'<path d="M{x - 2.3} 33H{x + 2.3}M{x} 30.7V35.3" stroke="{GREY}" stroke-width="1.1"/>'
        )
    t_blocks = [0.9, 1.05, 1.4, 1.55]
    for x, t in zip(blocks, t_blocks, strict=True):
        out.append(
            f'<rect x="{x}" y="27" width="18" height="12" rx="2" stroke="{ACC}" stroke-width="1.3" '
            f'fill="{ACC}" fill-opacity=".35" opacity="0">'
            + anim("opacity", pulse(S, t, 1))
            + "</rect>"
        )
    for x, t in zip(adds, (1.2, 1.7), strict=True):
        out.append(
            f'<g opacity="0"><circle cx="{x}" cy="33" r="4.5" stroke="{ACC}" stroke-width="1.4" fill="{ACC}" fill-opacity=".25"/>'
            f'<path d="M{x - 2.3} 33H{x + 2.3}M{x} 30.7V35.3" stroke="{ACC}" stroke-width="1.3"/>'
            + anim("opacity", pulse(S, t, 1))
            + "</g>"
        )
    for a, (t0, t1) in zip(arcs, ((0.9, 1.2), (1.4, 1.7)), strict=True):
        out.append(
            f'<path d="{a}" stroke="{ACC}" stroke-width="1.8" stroke-linecap="round" opacity="0">'
            + anim("opacity", pulse(S, t0, 1, ramp=0.1))
            + "</path>"
        )
        out.append(
            f'<g opacity="0"><circle r="4.5" fill="{ACC}" fill-opacity=".25"/><circle r="2.3" fill="{ACC}"/>'
            + motion(a, S, t0, t1)
            + anim(
                "opacity",
                [(S + t0, 0), (S + t0 + 0.03, 1), (S + t1, 1), (S + t1 + 0.06, 0)],
            )
            + "</g>"
        )
    return "".join(out)


def vit(S):
    """Return the 1-D ViT schematic of the phase starting at S."""
    out = []
    for j in range(4):
        out.append(
            f'<rect x="{12 + 16 * j + 0.5}" y="9" width="15" height="33" rx="2" fill="{ACC}" fill-opacity=".18" opacity="0">'
            + anim("opacity", pulse(S, 0.85 + 0.12 * j, 1))
            + "</rect>"
        )
    out.append(
        f'<path d="{mini_spec(12, 76, 40, 28, 5)}" stroke="{GREY}" stroke-width="1.1" stroke-linejoin="round"/>'
    )
    out.append(
        f'<path d="M28 9V42M44 9V42M60 9V42" stroke="{GREY}" stroke-width=".9" stroke-dasharray="2 2"/>'
    )
    out.append(arrow(81, 91, 25))
    cx = [102, 117, 132, 147]
    ty = 31
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    weights = [0.55, 0.9, 0.45, 0.75, 0.35, 0.95]

    def arc(i, j):
        h = {1: 6, 2: 11, 3: 16}[j - i]
        return f"M{cx[i]} {ty - 1}Q{(cx[i] + cx[j]) / 2:g} {ty - 1 - 2 * h} {cx[j]} {ty - 1}"

    out.append(
        f'<g stroke="{GREY}" stroke-width=".9" stroke-opacity=".55">'
        + "".join(f'<path d="{arc(i, j)}"/>' for i, j in pairs)
        + "</g>"
    )
    for m, ((i, j), w) in enumerate(zip(pairs, weights, strict=True)):
        out.append(
            f'<path d="{arc(i, j)}" stroke="{ACC}" stroke-width="{n(0.8 + 1.2 * w)}" stroke-linecap="round" opacity="0">'
            + anim("opacity", pulse(S, 1.45 + 0.08 * m, w))
            + "</path>"
        )
    for j, x in enumerate(cx):
        out.append(
            f'<rect x="{x - 4}" y="{ty}" width="8" height="8" rx="1.5" stroke="{GREY}" stroke-width="1" '
            f'fill="{GREY}" fill-opacity=".18"/>'
        )
        out.append(
            f'<rect x="{x - 4}" y="{ty}" width="8" height="8" rx="1.5" stroke="{ACC}" stroke-width="1.1" '
            f'fill="{ACC}" fill-opacity=".85" opacity="0">'
            + anim("opacity", pulse(S, 0.95 + 0.12 * j, 1))
            + "</rect>"
        )
    return "".join(out)


SCHEM = [mlp, cnn, resnet, vit]


def build() -> str:
    """Return the SVG document as a string."""
    parts = []
    P = parts.append

    P(
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="320" viewBox="0 0 800 320" '
        'role="img" aria-labelledby="t d" fill="none">'
    )
    P(
        '<title id="t">MaldiDeepKit: four deep-learning backbones, one scikit-learn API</title>'
    )
    P(
        '<desc id="d">Animated diagram. A binned MALDI-TOF spectrum (6000 bins, 2,000 to 20,000 m/z) '
        "is routed in turn through each of the four MaldiDeepKit backbones: an MLP (fully connected layers), "
        "a 1-D CNN (a convolution kernel sliding over the spectrum into feature maps), a 1-D ResNet "
        "(convolution blocks with skip connections) and a 1-D Vision Transformer (spectrum patches turned into "
        "tokens linked by self-attention). The active backbone is highlighted and its schematic animates, then "
        "the predicted antimicrobial-resistance probabilities update (illustrative values: MLP 0.82 Resistant / "
        "0.18 Susceptible, 1-D CNN 0.88 / 0.12, 1-D ResNet 0.91 / 0.09, 1-D ViT 0.86 / 0.14). All four share the "
        "same scikit-learn fit / predict_proba API. The animation loops every 12 seconds.</desc>"
    )
    P(
        '<style>text{font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}'
        f".tx{{fill:{TXT}}}.h{{font-size:11px;font-weight:600;letter-spacing:.08em}}"
        ".s{font-size:11px}.nm{font-size:14px;font-weight:600}.v{font-size:12.5px;font-weight:600}"
        ".lt{font-size:13px;font-weight:700}</style>"
    )
    P(
        f'<defs><clipPath id="cs"><rect x="{n(SX0)}" y="{n(SBASE - SH - 4)}" width="0" height="{n(SH + 6)}">'
    )
    sweep = []
    for k in range(4):
        S = k * PH
        sweep += [(S, 0), (S + 0.45, SX1 - SX0), (S + 0.9, SX1 - SX0), (S + 0.95, 0)]
    P(anim("width", sweep))
    P("</rect></clipPath>")
    P('<clipPath id="cf"><rect x="114" y="10" width="0" height="32">')
    S1 = 1 * PH
    P(anim("width", [(S1 + 0.85, 0), (S1 + 2.05, 42), (S1 + 2.95, 42), (S1 + 3.0, 0)]))
    P("</rect></clipPath></defs>")

    # column headers
    P(
        '<g class="tx h">'
        f'<text x="{n(SX0)}" y="26">INPUT</text>'
        f'<text x="{CX}" y="26">BACKBONE</text>'
        '<text x="624" y="26">OUTPUT</text></g>'
    )

    P(f'<text class="tx s" x="{n(SX0)}" y="104">6000-bin spectrum</text>')
    P(spec_bars(GREY))
    sw_op = []
    for k in range(4):
        S = k * PH
        sw_op += [(S + 0.55, 1), (S + 0.85, 0), (S + 0.97, 0), (S + 0.98, 1)]
    P('<g clip-path="url(#cs)">' + anim("opacity", sw_op) + spec_bars(ACC) + "</g>")
    P(
        f'<path d="M{n(SX0)} {n(SBASE + 0.5)}H{n(SX1)}" stroke="{GREY}" stroke-width="1"/>'
    )
    P(
        f'<text class="tx s" x="{n(SX0)}" y="201">2,000</text>'
        f'<text class="tx s" x="{n(SX1)}" y="201" text-anchor="end">20,000 m/z</text>'
    )

    P(f'<g stroke="{GREY}" stroke-width="1.2" stroke-opacity=".45">')
    for y in CY:
        yc = y + CH / 2
        P(f'<path d="{lpath(yc)}"/><path d="{rpath(yc)}"/>')
    P("</g>")
    for k, y in enumerate(CY):
        S = k * PH
        yc = y + CH / 2
        flow = '<animate attributeName="stroke-dashoffset" values="16;0" dur=".8s" repeatCount="indefinite"/>'
        P(
            f'<path d="{lpath(yc)}" stroke="{ACC}" stroke-width="2" stroke-dasharray="3 5" '
            f'stroke-linecap="round" opacity="0">{flow}'
            + anim("opacity", [(S + 0.3, 0), (S + 0.4, 1), (S + 2.85, 1), (S + 3.0, 0)])
            + "</path>"
        )
        P(
            f'<path d="{rpath(yc)}" stroke="{ACC}" stroke-width="2" stroke-dasharray="3 5" '
            f'stroke-linecap="round" opacity="0">{flow}'
            + anim(
                "opacity", [(S + 2.05, 0), (S + 2.15, 1), (S + 2.85, 1), (S + 3.0, 0)]
            )
            + "</path>"
        )
    # travelling pulses
    for k, y in enumerate(CY):
        S = k * PH
        yc = y + CH / 2
        for path, t0, t1 in ((lpath(yc), 0.35, 0.8), (rpath(yc), 2.05, 2.45)):
            P(
                f'<g opacity="0"><circle r="6" fill="{ACC}" fill-opacity=".22"/><circle r="3.2" fill="{ACC}"/>'
                + motion(path, S, t0, t1)
                + anim(
                    "opacity",
                    [(S + t0, 0), (S + t0 + 0.04, 1), (S + t1 - 0.04, 1), (S + t1, 0)],
                )
                + "</g>"
            )
    P(
        f'<circle cx="{HUB_L[0]}" cy="{HUB_L[1]}" r="3" fill="{GREY}"/>'
        f'<circle cx="{HUB_R[0]}" cy="{HUB_R[1]}" r="3" fill="{GREY}"/>'
    )

    for k, (y, (name, cls, _p)) in enumerate(zip(CY, MODELS, strict=True)):
        S = k * PH
        P(f'<g transform="translate({CX} {y})">')
        P(
            f'<rect x=".5" y=".5" width="{CW - 1}" height="{CH - 1}" rx="8" fill="{GREY}" fill-opacity=".07" '
            f'stroke="{GREY}" stroke-opacity=".45"/>'
        )
        P(
            f'<rect x=".5" y=".5" width="{CW - 1}" height="{CH - 1}" rx="8" fill="{ACC}" fill-opacity=".08" '
            f'stroke="{ACC}" stroke-width="1.6" opacity="0">'
            + anim("opacity", pulse(S, 0.6, 1, ramp=0.25))
            + "</rect>"
        )
        P(SCHEM[k](S))
        P(
            f'<text class="nm" x="176" y="23" fill="{TXT}">{name}'
            + anim("fill", colour_kf(S))
            + "</text>"
        )
        P(f'<text class="tx s" x="176" y="39">{cls}</text>')
        P("</g>")

    OX, BX, BW = 624, 640, 104
    P(f'<text class="tx s" x="{OX}" y="106">predict_proba(X)</text>')
    rows = [("R", "Resistant", R_C, 124), ("S", "Susceptible", S_C, 168)]
    dim = []
    for k in range(4):
        S = k * PH
        dim += [(S + 0.3, 1), (S + 0.5, 0.5), (S + 2.4, 0.5), (S + 2.55, 1)]
    for letter, label, col, yb in rows:
        P(f'<text class="lt" x="{OX}" y="{yb + 10.5}" fill="{col}">{letter}</text>')
        P(
            f'<rect x="{BX}" y="{yb}" width="{BW}" height="12" rx="3" fill="{GREY}" fill-opacity=".16"/>'
        )
        P(f'<text class="tx s" x="{BX}" y="{yb + 27}">{label}</text>')
    out_g = ['<g opacity="1">' + anim("opacity", dim)]
    for letter, _label, col, yb in rows:
        probs = [p if letter == "R" else 1 - p for _, _, p in MODELS]
        wk = [(0.0, probs[3] * BW)]
        for k in range(4):
            S = k * PH
            prev = probs[k - 1] * BW
            wk += [(S + 2.45, prev), (S + 2.6, 0.08 * BW), (S + 2.95, probs[k] * BW)]
        out_g.append(
            f'<rect x="{BX}" y="{yb}" width="{n(probs[0] * BW)}" height="12" rx="3" fill="{col}">'
            + anim("width", wk, spline=True)
            + "</rect>"
        )
        for k in range(4):
            S = k * PH
            if k < 3:
                kf = [(0, 0), (S + 2.55, 0), (S + 2.7, 1), (S + 5.45, 1), (S + 5.55, 0)]
            else:
                kf = [(0, 1), (2.45, 1), (2.55, 0), (S + 2.55, 0), (S + 2.7, 1)]
            out_g.append(
                f'<text class="v" x="784" y="{yb + 10.5}" text-anchor="end" fill="{col}" '
                f'opacity="{1 if k == 0 else 0}">{probs[k]:.2f}'
                + anim("opacity", kf)
                + "</text>"
            )
    for k, (name, _c, _p) in enumerate(MODELS):
        S = k * PH
        if k < 3:
            kf = [(0, 0), (S + 2.55, 0), (S + 2.7, 1), (S + 5.45, 1), (S + 5.55, 0)]
        else:
            kf = [(0, 1), (2.45, 1), (2.55, 0), (S + 2.55, 0), (S + 2.7, 1)]
        out_g.append(
            f'<text class="tx s" x="{OX}" y="218" opacity="{1 if k == 0 else 0}">via {name}'
            + anim("opacity", kf)
            + "</text>"
        )
    out_g.append("</g>")
    P("".join(out_g))

    cap = [
        f'<text x="400" y="298" text-anchor="middle" font-size="13" fill="{TXT}">'
        "<tspan>Same sklearn API, four backbones: </tspan>"
    ]
    for k, (name, _c, _p) in enumerate(MODELS):
        S = k * PH
        if k:
            cap.append("<tspan> · </tspan>")
        cap.append(
            f'<tspan font-weight="600">{name}' + anim("fill", colour_kf(S)) + "</tspan>"
        )
    cap.append("</text>")
    P("".join(cap))
    P("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    """Write the SVG to the path given on the command line, or to DEFAULT_OUT."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    svg = build()
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"Wrote {out} ({len(svg.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
