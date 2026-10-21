"""Shared pieces for "remove, then explain what remains" (see run_explain.py for the order).

The working state is a cleaned RGB image plus a removal mask. A stage that identifies a
feature (text, a border, a river, a legend symbol) erases it from the image, filling with the
local paper colour, and adds it to the mask. Later stages see a simpler map. Only gap tests
consult the mask: a gap inside removed ground means "something was here, can't tell", a gap
outside it is real paper. Erasing alone would turn every crossing into a false gap.
"""

import numpy as np
from scipy import ndimage as ndi

from common import SHEET  # noqa: E402,F401  (re-exported: other modules import it from here)


def rgb_to_lab(rgb):
    """sRGB (0-255, ... x 3) to CIELAB (D65). Colour distances are taken in Lab (delta E)."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ink_mask(rgb, alpha=None):
    """Ink = pixels clearly darker or more saturated than the local paper.

    The threshold comes from the sheet: paper is the bright mode of the lightness histogram, and
    ink is anything more than PAPER_GAP below it or more than CHROMA_GAP more saturated (so
    coloured inks count as ink on a coloured map, not just dark ones).
    """
    lab = rgb_to_lab(rgb)
    L, chroma = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    valid = np.ones(L.shape, bool) if alpha is None else alpha > 0
    hist, edges = np.histogram(L[valid], bins=100, range=(0, 100))
    paper_L = edges[np.argmax(hist)] + 0.5
    paper_c = np.median(chroma[valid & (np.abs(L - paper_L) < 3)])
    return valid & ((L < paper_L - 25) | (chroma > paper_c + 25)), paper_L


def paper_background(rgb, ink, sigma=12):
    """Local paper colour everywhere: a blur of the non-ink pixels only (normalized convolution).
    Filling erased ink with this keeps the paper's texture and tint, unlike a flat fill."""
    w = (~ink).astype(float)
    out = np.empty(rgb.shape, float)
    den = ndi.gaussian_filter(w, sigma) + 1e-6
    glob = np.median(rgb[~ink], axis=0)
    for k in range(3):
        out[..., k] = ndi.gaussian_filter(rgb[..., k] * w, sigma) / den
    far = den < 0.05  # deep inside a large inked area (a pool): no paper nearby to borrow from
    out[far] = glob
    return np.clip(out, 0, 255).astype(np.uint8)


class State:
    """The cleaned image, its ink, and what has been removed so far (by which stage)."""

    def __init__(self, rgb, ink, paper):
        self.rgb, self.ink, self.paper = rgb.copy(), ink.copy(), paper
        self.removed = np.zeros(ink.shape, bool)
        self.by = {}

    def remove(self, name, mask, grow=1):
        m = ndi.binary_dilation(mask, iterations=grow) & self.ink if grow else mask & self.ink
        self.rgb[m] = self.paper[m]
        self.ink &= ~m
        self.removed |= m
        self.by[name] = self.by.get(name, np.zeros_like(m)) | m
        return int(m.sum())
