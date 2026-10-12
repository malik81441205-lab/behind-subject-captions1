import numpy as np


def _over(base_f32, layer_rgba_u8):
    a = layer_rgba_u8[..., 3:4].astype(np.float32) / 255.0
    rgb = layer_rgba_u8[..., :3][..., ::-1].astype(np.float32)  # RGB -> BGR
    return base_f32 * (1 - a) + rgb * a


def composite(frame_bgr, mask, behind, front):
    """background -> text(behind) -> person -> text(front).
    The person is the original frame shown only where mask=1, so it covers the 'behind' text."""
    out = frame_bgr.astype(np.float32)
    if behind is not None:
        out = _over(out, behind)
    m = mask[..., None]
    out = out * (1 - m) + frame_bgr.astype(np.float32) * m
    if front is not None:
        out = _over(out, front)
    return np.clip(out, 0, 255).astype(np.uint8)


def find_head_top(mask, min_area=0.004, row_frac=0.02):
    """Top edge (pixel row) of the person in the mask, or None if nobody is clearly in frame."""
    h, w = mask.shape
    m = mask > 0.5
    if m.mean() < min_area:
        return None
    idx = np.nonzero(m.sum(axis=1) > row_frac * w)[0]
    return int(idx[0]) if len(idx) else None
