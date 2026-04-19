import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# ----------------------------
# Display
# ----------------------------
def show(title, img):
    plt.figure(figsize=(10, 6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# ----------------------------
# IO
# ----------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found")
    return img


# ----------------------------
# KMEANS segmentation
# ----------------------------
def kmeans_cluster_refined(img, K=5, spatial_weight=20, blur_size=9):
    img_blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2Lab)
    h, w = img_lab.shape[:2]
    color_features = img_lab.reshape((-1, 3)).astype(np.float32)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x / w).reshape(-1, 1).astype(np.float32)
    y = (y / h).reshape(-1, 1).astype(np.float32)

    Z = np.hstack([color_features, x * spatial_weight, y * spatial_weight])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    label_img = labels.reshape((h, w)).astype(np.uint8)
    return label_img.flatten(), (h, w), K


def get_cluster_masks(labels, shape, K):
    h, w = shape
    masks = []

    for i in range(K):
        m = (labels == i).astype(np.uint8).reshape(h, w) * 255

        _, comp, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if len(stats) <= 1:
            masks.append(m)
            continue

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(m)
        out[comp == largest] = 255

        masks.append(out)

    return masks


# ----------------------------
# Blue cluster detection
# ----------------------------
def find_blue_cluster(img, labels, K):
    best_idx, best_score = -1, -1

    for i in range(K):
        pixels = img.reshape(-1, 3)[labels == i]
        if len(pixels) == 0:
            continue

        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

        h = hsv[:, 0]
        s = hsv[:, 1]
        v = hsv[:, 2]

        mask = (h >= 90) & (h <= 140) & (s > 60) & (v > 40)

        blue_ratio = np.sum(mask) / len(mask + 1e-6)

        saturation_score = np.mean(s[mask]) / 255 if np.sum(mask) > 0 else 0
        purity = blue_ratio

        size_factor = len(pixels)

        score = purity * saturation_score * np.log1p(size_factor)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx



def select_target_cluster(img, labels, masks, blue_idx):

    kernel = np.ones((5, 5), np.uint8)

    blue = cv2.morphologyEx(masks[blue_idx], cv2.MORPH_OPEN, kernel, iterations=1)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=2)

    blue_near = cv2.dilate(blue, kernel, iterations=2)

    inv_blue = (blue == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv_blue, cv2.DIST_L2, 5)

    best_i, best_score = -1, -1

    for i, mask in enumerate(masks):
        if i == blue_idx:
            continue

        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num, comp, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            m = np.where(comp == largest, 255, 0).astype(np.uint8)

        size = np.sum(m) / 255
        if size < 1000:
            continue

        overlap = np.sum(cv2.bitwise_and(m, blue_near)) / 255
        proximity = overlap / (size + 1e-6)

        coords = np.column_stack(np.where(m > 0))
        dists = dist_map[coords[:, 0], coords[:, 1]]
        dist_score = 1.0 / (np.median(dists) + 1e-6)

        components = cv2.connectedComponents(m)[0]
        frag_penalty = 1.0 / (components + 1e-6)

        score = (
            0.55 * proximity +
            0.30 * dist_score +
            0.15 * frag_penalty
        )

        if score > best_score:
            best_score, best_i = score, i

    return best_i


# ----------------------------
# Line merging
# ----------------------------
def merge_lines(lines, angle_thresh=0.3, dist_thresh=210):

    merged = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        m, b = ((y2 - y1)/(x2 - x1), y1 - ((y2 - y1)/(x2 - x1))*x1) if x2 != x1 else (np.inf, x1)

        found = False
        for idx, (mr, br, x1r, y1r, x2r, y2r) in enumerate(merged):
            if (m != np.inf and mr != np.inf and abs(m - mr) < angle_thresh and abs(b - br) < dist_thresh) or \
               (m == np.inf and mr == np.inf and abs(b - br) < dist_thresh):
                xs, ys = [x1, x2, x1r, x2r], [y1, y2, y1r, y2r]
                merged[idx] = (mr, br, min(xs), min(ys), max(xs), max(ys))
                found = True
                break
        if not found:
            merged.append((m, b, x1, y1, x2, y2))
    return merged



def intersect(l1, l2):
    m1, b1, x1, y1, x2, y2 = l1
    m2, b2, x3, y3, x4, y4 = l2

    if m1 == np.inf and m2 == np.inf:
        return None

    if m1 == np.inf:
        px = b1
        py = m2 * px + b2
    elif m2 == np.inf:
        px = b2
        py = m1 * px + b1
    elif abs(m1 - m2) < 1e-4:
        return None
    else:
        px = (b2 - b1) / (m1 - m2)
        py = m1 * px + b1

    return (px, py)



def compute_corners_from_lines(lines):
    if len(lines) != 4:
        return None

    data = []
    for l in lines:
        m = l[0]
        angle = np.degrees(np.arctan(m)) if m != np.inf else 90
        data.append((angle, l))

    data.sort(key=lambda x: x[0])

    pairs = [
        abs(data[0][0] - data[1][0]),
        abs(data[1][0] - data[2][0]),
        abs(data[2][0] - data[3][0]),
        180 - abs(data[0][0] - data[3][0])
    ]

    idx = np.argmin(pairs)

    if idx == 0:
        A, B = [data[0][1], data[1][1]], [data[2][1], data[3][1]]
    elif idx == 1:
        A, B = [data[1][1], data[2][1]], [data[0][1], data[3][1]]
    elif idx == 2:
        A, B = [data[2][1], data[3][1]], [data[0][1], data[1][1]]
    else:
        A, B = [data[3][1], data[0][1]], [data[1][1], data[2][1]]

    pts = []
    for a in A:
        for b in B:
            p = intersect(a, b)
            if p:
                pts.append(p)

    return pts if len(pts) == 4 else None



def get_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)



def order_points(pts):
    pts.sort(key=lambda c: c[1])
    tl = pts[0] if get_dist(pts[0], [0, pts[0][1]]) < get_dist(pts[1], [0, pts[1][1]]) else pts[1]
    tr = pts[1] if np.array_equal(tl, pts[0]) else pts[0]
    bl = pts[2] if get_dist(pts[2], [0, pts[2][1]]) < get_dist(pts[3], [0, pts[3][1]]) else pts[3]
    br = pts[3] if np.array_equal(bl, pts[2]) else pts[2]
    return np.array([tl, tr, br, bl], dtype="float32")


# ----------------------------
# Warp
# ----------------------------
def warp_from_cluster(img, mask, return_meta=False):
    h, w = img.shape[:2]
    PAD = int(0.5 * max(h, w))

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=3
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, None) if return_meta else None

    outer = max(contours, key=cv2.contourArea)
    outer_mask = np.zeros_like(mask)
    cv2.drawContours(outer_mask, [outer], -1, 255, 3)

    padded = cv2.copyMakeBorder(outer_mask, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, 0)

    edges = cv2.Canny(padded, 100, 200)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=140, maxLineGap=200)

    if lines is None:
        return (None, None) if return_meta else None

    line_img = np.zeros_like(edges)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

    shifted = [np.array([[x1 - PAD, y1 - PAD, x2 - PAD, y2 - PAD]]) for x1, y1, x2, y2 in lines[:, 0]]

    merged = merge_lines(shifted)
    merged = sorted(merged, key=lambda l: (np.hypot(l[5]-l[3], l[4]-l[2]), abs(l[0])), reverse=True)[:4]

    if len(merged) < 4:
        return (None, None) if return_meta else None

    line_img = np.zeros_like(edges)
    for m, b, x1, y1, x2, y2 in merged:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

    corners = compute_corners_from_lines(merged)
    if not corners:
        return (None, None) if return_meta else None

    src = order_points(corners) + PAD

    dst = np.array([[0, 800], [0, 0], [1600, 0], [1600, 800]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    padded_img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_temp = cv2.warpPerspective(padded_img, M, (1600, 800))
    if np.count_nonzero(img_temp) < (img_temp.size / 2):
        return (None, None) if return_meta else None

    if not return_meta:
        return img_temp

    meta = {
        "M": M,
        "M_inv": np.linalg.inv(M),
        "src": src,
        "dst": dst,
        "pad": PAD,
        "warped_shape": img_temp.shape[:2],
    }
    return img_temp, meta



def extract_blue_cloth(warped_img, padding=50, return_bbox=False):
    if warped_img is None:
        return (None, None) if return_bbox else None

    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if return_bbox:
            h, w = warped_img.shape[:2]
            return warped_img, {"x": 0, "y": 0, "width": w, "height": h}
        return warped_img

    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)

    img_h, img_w = warped_img.shape[:2]

    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    w_new = min(img_w - x_new, w + (2 * padding))
    h_new = min(img_h - y_new, h + (2 * padding))

    crop = warped_img[y_new: y_new + h_new, x_new: x_new + w_new]
    crop_bbox = {"x": int(x_new), "y": int(y_new), "width": int(w_new), "height": int(h_new)}

    return (crop, crop_bbox) if return_bbox else crop


# ----------------------------
# Ball number mapping
# ----------------------------
BALL_NUMBER_MAP = {
    "yellow": (1, 9),
    "blue":   (2, 10),
    "red":    (3, 11),
    "purple": (4, 12),
    "orange": (5, 13),
    "green":  (6, 14),
    "maroon": (7, 15),
}


# ----------------------------
# Classification (NEW, robust)
# ----------------------------
def estimate_cloth_lab_global(img_bgr):
    """Median LAB of felt pixels (blue cloth)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    cloth_m = (
        (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 115) &
        (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 80)
    )
    if np.count_nonzero(cloth_m) > 1000:
        return np.median(lab[cloth_m], axis=0).astype(np.float32)
    return None


def classify_ball_final(img_bgr, cx, cy, r, cloth_lab=None):
    """
    Robust ball classification.

    Strategy:
      1) Test 8-ball and cue-ball on the INNER disk (0.65*r) BEFORE background
         filtering — this is decisive for black (low S,V) and white (low S, high V)
         balls, and avoids confusion with the felt that bleeds into the full disk.
      2) Otherwise, subtract the cloth colour via delta_LAB and analyse only
         foreground pixels.
      3) Stripe vs solid: ratio of white pixels on the foreground.
      4) Colour: hue bins on the top-saturated 40% of foreground pixels, with
         blue/purple discriminated by the LAB b* channel (how 'blue' the ball is).

    Returns: (number, color_name, is_stripe) or (None, None, False).
    """
    H, W = img_bgr.shape[:2]
    pad = int(r * 1.15)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(W, cx + pad + 1), min(H, cy + pad + 1)

    roi_bgr = img_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None, None, False

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.int32)

    hH, hW = roi_hsv.shape[:2]
    yy, xx = np.ogrid[:hH, :hW]
    lx, ly = cx - x1, cy - y1

    disk_inner = (xx - lx) ** 2 + (yy - ly) ** 2 <= (0.65 * r) ** 2
    disk_full  = (xx - lx) ** 2 + (yy - ly) ** 2 <= (0.95 * r) ** 2

    if np.count_nonzero(disk_full) < 15 or np.count_nonzero(disk_inner) < 8:
        return None, None, False

    # ===== 8-BALL / CUE check on INNER disk =====
    V_in = roi_hsv[:, :, 2][disk_inner].astype(np.int32)
    S_in = roi_hsv[:, :, 1][disk_inner].astype(np.int32)
    H_in = roi_hsv[:, :, 0][disk_inner].astype(np.int32)
    inner_total = V_in.size

    # "True black": low V AND low S (dark green has high S, so not confused)
    black_inner = (V_in < 80) & (S_in < 90)
    br_inner = np.count_nonzero(black_inner) / inner_total

    # Felt presence inside inner disk (bad-centred detection)
    cloth_inner = (H_in >= 95) & (H_in <= 115) & (S_in >= 150)
    cloth_inner_ratio = np.count_nonzero(cloth_inner) / inner_total

    if br_inner >= 0.55 and cloth_inner_ratio < 0.20:
        return 8, "black", False

    white_inner = (V_in > 160) & (S_in < 70)
    wr_inner = np.count_nonzero(white_inner) / inner_total
    if wr_inner >= 0.55 and np.median(S_in) < 95 and cloth_inner_ratio < 0.20:
        return 0, "white", False

    # ===== Background subtraction =====
    if cloth_lab is None:
        ring = ((xx - lx) ** 2 + (yy - ly) ** 2 <= (1.25 * r) ** 2) & ~disk_full
        if np.count_nonzero(ring) >= 10:
            cloth_lab = np.median(roi_lab[ring], axis=0)
        else:
            cloth_lab = np.array([120, 128, 100])

    delta = np.linalg.norm(roi_lab - cloth_lab, axis=2)
    fg_mask = disk_full & (delta > 22)
    if np.count_nonzero(fg_mask) < 20:
        fg_mask = disk_full & (delta > 14)
        if np.count_nonzero(fg_mask) < 12:
            return None, None, False

    H_p = roi_hsv[:, :, 0][fg_mask].astype(np.int32)
    S_p = roi_hsv[:, :, 1][fg_mask].astype(np.int32)
    V_p = roi_hsv[:, :, 2][fg_mask].astype(np.int32)
    L_p = roi_lab[:, :, 0][fg_mask]
    a_p = roi_lab[:, :, 1][fg_mask]
    b_p = roi_lab[:, :, 2][fg_mask]
    total = H_p.size

    white_m = (S_p < 55) & (V_p > 160)
    black_m = (V_p < 65) & (S_p < 100)
    color_m = ~white_m & ~black_m & (S_p > 55)

    wr = np.count_nonzero(white_m) / total
    br = np.count_nonzero(black_m) / total
    cr = np.count_nonzero(color_m) / total

    # Fallbacks for ambiguous cases after background subtraction
    if wr >= 0.50 and cr < 0.30 and br < 0.10:
        return 0, "white", False
    if br >= 0.55 and cr < 0.20:
        return 8, "black", False

    if np.count_nonzero(color_m) < 12:
        if wr > br and wr > 0.30:
            return 0, "white", False
        if br > 0.30:
            return 8, "black", False
        return None, None, False

    # ===== Colour analysis on top-saturated pixels =====
    s_c = S_p[color_m]
    v_c = V_p[color_m]
    h_c = H_p[color_m]
    a_c = a_p[color_m]
    b_c = b_p[color_m]
    L_c = L_p[color_m]

    if len(s_c) >= 25:
        thr = np.percentile(s_c, 60)
        sel = s_c >= thr
        if np.count_nonzero(sel) < 8:
            sel = np.ones_like(s_c, dtype=bool)
    else:
        sel = np.ones_like(s_c, dtype=bool)

    h_sel = h_c[sel]
    s_sel = s_c[sel]
    v_sel = v_c[sel]
    a_sel = a_c[sel]
    b_sel = b_c[sel]
    L_sel = L_c[sel]

    # Red wrap-around (H near 0 and 180)
    if np.percentile(h_sel, 10) < 5 and np.percentile(h_sel, 90) > 170:
        h_adj = np.where(h_sel > 90, h_sel.astype(np.int32) - 180, h_sel.astype(np.int32))
        mh = int(np.median(h_adj)) % 180
    else:
        mh = int(np.median(h_sel))
    ms = int(np.median(s_sel))
    mv = int(np.median(v_sel))
    mL = int(np.median(L_sel))
    ma = int(np.median(a_sel))
    mb = int(np.median(b_sel))

    # Stripe detection: enough white pixels AND enough colour pixels
    is_stripe = (wr >= 0.14 and cr >= 0.25) or \
                (wr >= 0.10 and cr >= 0.20 and (wr + br) >= 0.25)

    color_name = None

    # RED (3/11) — low hue, very saturated, strong LAB a*
    if (mh <= 8 or mh >= 172) and ms >= 140 and ma >= 150:
        color_name = "red"
    # ORANGE (5/13) — hue 8-17, saturated, bright
    elif 8 <= mh <= 17 and ms >= 130 and mv >= 165:
        color_name = "orange"
    # MAROON dark (7/15) — low hue but dark
    elif mh <= 16 and mv < 165:
        color_name = "maroon"
    # YELLOW (1/9) — hue 16-35, bright AND saturated, b* high
    elif 16 <= mh <= 35 and mv >= 165 and mb >= 155 and ms >= 130:
        color_name = "yellow"
    # MAROON faded / YELLOW with broken conditions fallback in hue 16-35
    elif 16 <= mh <= 35:
        color_name = "yellow" if (mb >= 170 and mv >= 180) else "maroon"
    # GREEN (6/14)
    elif 35 <= mh <= 95 and ms >= 55:
        color_name = "green"
    # BLUE vs PURPLE (95-145) — decide by LAB b* (how blue)
    elif 95 <= mh <= 145:
        if mb <= 107:
            color_name = "blue"
        elif mb >= 112:
            color_name = "purple"
        else:
            color_name = "blue" if (ms >= 190 and mv < 110) else "purple"
    else:
        REFS_LAB = {
            "yellow": (180, 140, 180), "blue":   (70, 130,  98),
            "red":    (95, 170, 150),  "purple": (80, 130, 108),
            "orange": (135, 155, 165), "green":  (115, 108, 135),
            "maroon": (125, 140, 150),
        }
        mean_lab = np.array([mL, ma, mb], dtype=np.float32)
        dists = {n: float(np.linalg.norm(mean_lab - np.array(ref))) for n, ref in REFS_LAB.items()}
        color_name = min(dists, key=dists.get)

    number = BALL_NUMBER_MAP[color_name][1 if is_stripe else 0]
    return int(number), color_name, is_stripe


# ----------------------------
# Ball detection on original image
# ----------------------------
def _detect_balls_on_original(img, blue_mask):
    H, W = img.shape[:2]

    filled = np.zeros_like(blue_mask)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)

    ys, xs = np.where(filled > 0)
    if len(xs) == 0:
        return []
    table_w = int(xs.max() - xs.min())
    min_r = max(10, int(table_w * 0.012))
    max_r = max(min_r + 5, int(table_w * 0.025))

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    cloth_lab_local = np.median(lab_img[filled > 0], axis=0)
    delta_lab = np.linalg.norm(lab_img - cloth_lab_local, axis=2)

    gray_blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    all_raw = []
    for p1, p2, dp in [(50, 12, 1.0), (60, 15, 1.0), (40, 10, 1.1),
                       (70, 18, 1.0), (50, 10, 1.2), (45, 11, 1.0)]:
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=dp,
                                   minDist=int(1.8 * min_r), param1=p1, param2=p2,
                                   minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            for x, y, r in np.round(circles[0]).astype(int):
                if 0 <= y < H and 0 <= x < W and filled[y, x] > 0:
                    all_raw.append((int(x), int(y), int(r)))

    scored = []
    for x, y, r in all_raw:
        yy, xx = np.ogrid[:H, :W]
        disk_in = ((xx - x) ** 2 + (yy - y) ** 2 <= (0.8 * r) ** 2) & (filled > 0)
        if np.count_nonzero(disk_in) < 10:
            continue
        scored.append((float(np.mean(delta_lab[disk_in])), x, y, r))

    scored.sort(key=lambda c: -c[0])
    detected = []
    for sc, x, y, r in scored:
        if sc < 30:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (max(r, pr) * 1.6) ** 2 for _, px, py, pr in detected):
            continue
        detected.append((sc, x, y, r))

    # Global cloth reference (more robust than local ring)
    cloth_lab_global = estimate_cloth_lab_global(img)

    balls = []
    for sc, x, y, r in detected:
        num, color_name, is_stripe = classify_ball_final(img, x, y, r, cloth_lab=cloth_lab_global)
        if num is None:
            continue
        balls.append({
            "number": int(num),
            "color": color_name,
            "stripe": bool(is_stripe),
            "bbox": {"x": int(x - r), "y": int(y - r), "width": int(2 * r), "height": int(2 * r)},
            "center": [int(x), int(y)],
            "radius": int(r),
        })

    return balls


# ----------------------------
# Drawing
# ----------------------------
def draw_ball_detections_normalized(img, balls, W, H):
    out = img.copy()
    for b in balls:
        x1 = int(b["xmin"] * W); y1 = int(b["ymin"] * H)
        x2 = int(b["xmax"] * W); y2 = int(b["ymax"] * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, str(b["number"]), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, str(b["number"]), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(image_path, display=False):
    img = load_image(image_path)
    H, W = img.shape[:2]

    labels, shape, K = kmeans_cluster_refined(img, 5)
    masks = get_cluster_masks(labels, shape, K)
    blue_idx = find_blue_cluster(img, labels, K)

    if blue_idx == -1:
        return {"image_path": image_path, "num_balls": 0, "balls": [], "top_view": None, "img": img}

    blue_mask = masks[blue_idx]
    balls = _detect_balls_on_original(img, blue_mask)

    output_balls = []
    for b in balls:
        bx, by, bw, bh = b["bbox"]["x"], b["bbox"]["y"], b["bbox"]["width"], b["bbox"]["height"]
        output_balls.append({
            "number": b["number"],
            "xmin": float(max(0, bx) / W),
            "xmax": float(min(W, bx + bw) / W),
            "ymin": float(max(0, by) / H),
            "ymax": float(min(H, by + bh) / H),
        })

    target = select_target_cluster(img, labels, masks, blue_idx)
    warped = warp_from_cluster(img, masks[target]) if target != -1 else None
    top_view = extract_blue_cloth(warped, padding=20) if warped is not None else None

    if display:
        show("Original", img)
        if top_view is not None:
            show("Top View", top_view)
        print(f"Total balls: {len(output_balls)}")
        for b in output_balls:
            print(f"  Ball {b['number']}")

    return {
        "image_path": image_path,
        "num_balls": len(output_balls),
        "balls": output_balls,
        "top_view": top_view,
        "img": img,
    }


def process_images(input_json, output_json, top_view_dir=None, annotated_dir=None, display=False):
    with open(input_json, "r") as f:
        data = json.load(f)
    image_paths = data.get("image_path", [])

    if top_view_dir:
        os.makedirs(top_view_dir, exist_ok=True)
    if annotated_dir:
        os.makedirs(annotated_dir, exist_ok=True)

    results = []
    for path in image_paths:
        print(f"Processing {path}...")
        try:
            result = run_pipeline(path, display=display)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {"image_path": path, "num_balls": 0, "balls": [], "top_view": None, "img": None}

        results.append({
            "image_path": result["image_path"],
            "num_balls": result["num_balls"],
            "balls": result["balls"],
        })

        name = os.path.basename(path)

        if top_view_dir and result.get("top_view") is not None:
            cv2.imwrite(os.path.join(top_view_dir, name), result["top_view"])

        if annotated_dir and result.get("img") is not None:
            img = result["img"]
            H_img, W_img = img.shape[:2]
            annotated = draw_ball_detections_normalized(img, result["balls"], W_img, H_img)
            cv2.imwrite(os.path.join(annotated_dir, name), annotated)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output_json}")
    return results


if __name__ == "__main__":
    BASE = r"C:\Users\victo\OneDrive - Universidade do Porto\MIA_1Y2S\VC_2"
    development_dir = os.path.join(BASE, "development_set")

    process_images(
        input_json=os.path.join(BASE, "input_test.json"),
        output_json=os.path.join(BASE, "output.json"),
        top_view_dir=os.path.join(development_dir, "top_views"),
        annotated_dir=os.path.join(development_dir, "detections"),
    )
