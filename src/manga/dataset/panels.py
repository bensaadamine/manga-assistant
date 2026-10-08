"""Panel extraction from manga pages: recursive X-Y gutter cut."""
import numpy as np
import cv2


def to_ink(gray, thr=200):
    """Binary mask, 1 = ink (dark)."""
    return (gray < thr).astype(np.uint8)


def trim(ink, x0, y0, x1, y1, max_ink_frac=0.01):
    """Shrink a box inward past fully-white rows/cols."""
    sub = ink[y0:y1, x0:x1]
    if sub.size == 0:
        return x0, y0, x1, y1
    h, w = sub.shape
    cols = sub.sum(0) > max_ink_frac * h
    rows = sub.sum(1) > max_ink_frac * w
    if not cols.any() or not rows.any():
        return x0, y0, x0, y0
    cx = np.flatnonzero(cols)
    cy = np.flatnonzero(rows)
    return x0 + cx[0], y0 + cy[0], x0 + cx[-1] + 1, y0 + cy[-1] + 1


def _runs(mask):
    """Start/end indices of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def best_cut(ink, x0, y0, x1, y1, min_gutter, max_ink_frac, thin_gutter,
             thin_ink_frac, min_part_ink, line_ink_frac):
    """Widest interior gutter in either axis.

    A gutter qualifies either as wide-and-mostly-white, or as thin-and-almost-
    perfectly-white. Manga gutters can be only a few px even on a 3440px page,
    but a 3px column that is 99.6% white across a whole panel height is still a
    far stronger signal than anything that occurs inside artwork.
    """
    sub = ink[y0:y1, x0:x1]
    cands = []
    for axis in (0, 1):
        if axis == 0:                      # vertical gutter -> split on x
            proj, span, n = sub.sum(0), sub.shape[0], sub.shape[1]
        else:                              # horizontal gutter -> split on y
            proj, span, n = sub.sum(1), sub.shape[1], sub.shape[0]
        seen = set()
        for thr, wmin in ((max_ink_frac, min_gutter), (thin_ink_frac, thin_gutter)):
            for a, b in _runs(proj <= thr * span):
                if a == 0 or b == n:       # edge margin, not an interior gutter
                    continue
                if b - a < wmin or (axis, a, b) in seen:
                    continue
                seen.add((axis, a, b))
                cands.append((b - a, axis, (a + b) // 2))
    cands.sort(reverse=True)

    # Panel borders as a fallback. Where a bubble or an SFX mark overhangs the
    # gutter there is no clean white band to cut on, but the drawn border line
    # is still there: a row or column that is almost entirely ink.
    lines = []
    for axis in (0, 1):
        if axis == 0:
            proj, span, n = sub.sum(0), sub.shape[0], sub.shape[1]
        else:
            proj, span, n = sub.sum(1), sub.shape[1], sub.shape[0]
        thick = max(2, int(0.012 * n))
        for a, b in _runs(proj >= line_ink_frac * span):
            if a == 0 or b == n or not (2 <= b - a <= thick):
                continue
            # a drawn border is a thin dark line with lighter content either
            # side; the edge of a black fill is dark on one side and must not
            # be mistaken for one.
            gap = 2 * (b - a) + 2
            lo = proj[max(0, a - gap):a]
            hi = proj[b:b + gap]
            if lo.size == 0 or hi.size == 0:
                continue
            if lo.min() > 0.45 * span or hi.min() > 0.45 * span:
                continue
            lines.append((b - a, axis, (a + b) // 2))
    lines.sort(reverse=True)

    # Reject a cut that would carve off a near-empty strip. Borderless panels
    # drawn on open white space otherwise shatter into dozens of fragments,
    # because every band of white between two sparse figures looks like a gutter.
    for width, axis, pos in cands + lines:
        if axis == 0:
            lo, hi = sub[:, :pos], sub[:, pos:]
        else:
            lo, hi = sub[:pos, :], sub[pos:, :]
        if lo.size == 0 or hi.size == 0:
            continue
        if min(lo.shape) < 8 or min(hi.shape) < 8:
            continue
        if min(lo.mean(), hi.mean()) < min_part_ink:
            continue
        return axis, pos, width
    return None


def split(ink, box, cuts, min_side, depth=0, max_depth=14):
    x0, y0, x1, y1 = trim(ink, *box)
    if x1 - x0 < min_side or y1 - y0 < min_side or depth >= max_depth:
        return [(x0, y0, x1, y1)] if x1 > x0 and y1 > y0 else []
    cut = best_cut(ink, x0, y0, x1, y1, **cuts)
    if cut is None:
        return [(x0, y0, x1, y1)]
    axis, pos, _ = cut
    if axis == 0:
        a = (x0, y0, x0 + pos, y1)
        b = (x0 + pos, y0, x1, y1)
    else:
        a = (x0, y0, x1, y0 + pos)
        b = (x0, y0 + pos, x1, y1)
    out = []
    for sb in (a, b):
        out += split(ink, sb, cuts, min_side, depth + 1, max_depth)
    return out


def detect(gray, gutter_frac=0.004, max_ink_frac=0.02, thin_frac=0.0012,
           thin_ink_frac=0.004, min_side_frac=0.07, min_part_ink=0.03,
           line_ink_frac=0.90):
    """Return panel boxes (x0,y0,x1,y1) for a grayscale page."""
    h, w = gray.shape
    ink = to_ink(gray)
    cuts = dict(min_gutter=max(5, int(gutter_frac * h)),
                max_ink_frac=max_ink_frac,
                thin_gutter=max(3, int(thin_frac * h)),
                thin_ink_frac=thin_ink_frac,
                min_part_ink=min_part_ink,
                line_ink_frac=line_ink_frac)
    return split(ink, (0, 0, w, h), cuts, max(40, int(min_side_frac * h)))


def draw(gray, boxes):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), max(2, gray.shape[0] // 500))
        cv2.putText(vis, str(i), (x0 + 8, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX,
                    gray.shape[0] / 2000, (255, 0, 0), 3)
    return vis
