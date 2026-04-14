import cv2
import numpy as np
import json
import sys
import os


FULL_TABLE_WIDTH = 1000
FULL_TABLE_HEIGHT = 500


def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def order_quad_tl_tr_br_bl(pts):
    pts = np.asarray(pts, dtype=np.float32)

    idx = np.argsort(pts[:, 1])
    top = pts[idx[:2]]
    bot = pts[idx[2:]]

    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]

    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def line_angle(line):
    a, b, _ = line
    ang = np.arctan2(-a, b)
    if ang < 0:
        ang += np.pi
    return ang


def angle_diff_pi(a, b):
    d = abs(a - b)
    return min(d, np.pi - d)


def fit_line(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    vx = float(vx[0])
    vy = float(vy[0])
    x0 = float(x0[0])
    y0 = float(y0[0])

    a = -vy
    b = vx
    c = vy * x0 - vx * y0

    s = np.sqrt(a * a + b * b) + 1e-8
    return np.array([a / s, b / s, c / s], dtype=np.float32)


def robust_fit_line(points, max_iter=5, inlier_thresh=2.5):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return None

    cur = pts.copy()
    for _ in range(max_iter):
        line = fit_line(cur)
        if line is None:
            return None

        a, b, c = line
        d = np.abs(a * cur[:, 0] + b * cur[:, 1] + c)
        keep = d < inlier_thresh

        if np.sum(keep) < 2:
            break
        cur = cur[keep]

    if len(cur) < 2:
        return None

    return fit_line(cur)


def intersect_lines(line1, line2):
    if line1 is None or line2 is None:
        return None

    a1, b1, c1 = line1
    a2, b2, c2 = line2

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None

    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)


def line_side_normal(p1, p2):
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    v = p2 - p1
    n = np.array([-v[1], v[0]], dtype=np.float32)
    s = np.linalg.norm(n) + 1e-8
    return n / s


def line_from_two_points(p1, p2):
    return fit_line(np.array([p1, p2], dtype=np.float32))


def point_line_distance(pt, line):
    a, b, c = line
    x, y = float(pt[0]), float(pt[1])
    return abs(a * x + b * y + c) / (np.sqrt(a * a + b * b) + 1e-8)


def line_direction_from_abc(line):
    a, b, _ = line
    d = np.array([b, -a], dtype=np.float32)
    n = np.linalg.norm(d) + 1e-8
    return d / n


def signed_point_line_value(pt, line):
    a, b, c = line
    x, y = float(pt[0]), float(pt[1])
    return a * x + b * y + c


def collect_points_in_disk(pts_all, center_pt, radius=70):
    pts_all = np.asarray(pts_all, dtype=np.float32)
    c = np.asarray(center_pt, dtype=np.float32)
    d = np.linalg.norm(pts_all - c[None, :], axis=1)
    out = pts_all[d <= radius]
    if len(out) < 2:
        return None
    return out


def side_lengths_of_quad(q):
    q = np.asarray(q, dtype=np.float32)
    lens = []
    for i in range(4):
        lens.append(float(np.linalg.norm(q[(i + 1) % 4] - q[i])))
    return lens


def collect_points_near_segment_trimmed(pts_all, p1, p2, dist_thresh=8.0, ext_frac=0.10, trim_t=0.12):
    pts_all = np.asarray(pts_all, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)

    v = p2 - p1
    vlen = np.linalg.norm(v)
    if vlen < 1e-6:
        return None

    u = v / vlen
    n = np.array([-u[1], u[0]], dtype=np.float32)

    w = pts_all - p1[None, :]
    proj = w @ u
    perp = np.abs(w @ n)

    lo = trim_t * vlen - ext_frac * vlen
    hi = (1.0 - trim_t) * vlen + ext_frac * vlen

    keep = (perp <= dist_thresh) & (proj >= lo) & (proj <= hi)
    out = pts_all[keep]

    if len(out) < 2:
        return None
    return out


def keep_best_table_component(mask):
    h, w = mask.shape[:2]
    cx, cy = w * 0.5, h * 0.5

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    best_score = -1e18
    best_label = None

    for i in range(1, num_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 200:
            continue

        x = float(centroids[i][0])
        y = float(centroids[i][1])
        bw = float(stats[i, cv2.CC_STAT_WIDTH])
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])

        box_area = max(bw * bh, 1.0)
        fill = area / box_area
        dist2 = (x - cx) ** 2 + (y - cy) ** 2

        score = area + 15000.0 * fill - 0.20 * dist2

        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        return None

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def component_containing_seed(mask, seed_mask):
    if mask is None:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    seed_bool = seed_mask > 0
    best_label = None
    best_overlap = 0

    for i in range(1, num_labels):
        overlap = int(np.sum((labels == i) & seed_bool))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = i

    if best_label is None or best_overlap == 0:
        return None

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def run_kmeans_labels(img, k=5, target_w=360):
    h, w = img.shape[:2]
    scale = target_w / float(w)
    small_w = target_w
    small_h = max(100, int(round(h * scale)))

    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    lab_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB)

    L = lab_small[:, :, 0].astype(np.float32)
    A = lab_small[:, :, 1].astype(np.float32)
    B = lab_small[:, :, 2].astype(np.float32)

    feats = np.stack([0.35 * L, 1.00 * A, 1.20 * B], axis=-1).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)

    _, labels, centers = cv2.kmeans(feats, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels_2d = labels.reshape(small_h, small_w)

    return img_small, labels_2d, centers


def build_cluster_component_masks(labels_2d):
    h, w = labels_2d.shape[:2]
    k = int(labels_2d.max()) + 1
    cluster_components = []

    for i in range(k):
        raw = np.zeros((h, w), dtype=np.uint8)
        raw[labels_2d == i] = 255

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_open, iterations=1)

        comp = keep_best_table_component(clean)
        cluster_components.append(comp)

    return cluster_components


def score_cluster_as_cloth(img_small, component_mask):
    if component_mask is None:
        return -1e18

    ys, xs = np.where(component_mask > 0)
    if len(xs) < 300:
        return -1e18

    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    h_vals = H[component_mask > 0]
    s_vals = S[component_mask > 0]
    v_vals = V[component_mask > 0]

    h_med = float(np.median(h_vals))
    s_med = float(np.median(s_vals))
    v_med = float(np.median(v_vals))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
    if num_labels <= 1:
        return -1e18

    area = float(stats[1, cv2.CC_STAT_AREA])
    cx = float(centroids[1][0])
    cy = float(centroids[1][1])
    bw = float(stats[1, cv2.CC_STAT_WIDTH])
    bh = float(stats[1, cv2.CC_STAT_HEIGHT])

    fill = area / max(1.0, bw * bh)
    dist2 = (cx - img_small.shape[1] * 0.5) ** 2 + (cy - img_small.shape[0] * 0.5) ** 2

    hue_score = max(0.0, 1.0 - abs(h_med - 100.0) / 28.0)
    sat_score = np.clip((s_med - 55.0) / 90.0, 0.0, 1.0)
    val_score = np.clip((v_med - 60.0) / 140.0, 0.0, 1.0)

    color_score = 3.0 * hue_score + 2.0 * sat_score + 1.0 * val_score
    return 28000.0 * color_score + area + 12000.0 * fill - 0.18 * dist2


def select_cloth_component(img_small, cluster_components):
    best_score = -1e18
    best_idx = -1
    best_mask = None

    for i, comp in enumerate(cluster_components):
        score = score_cluster_as_cloth(img_small, comp)
        if score > best_score:
            best_score = score
            best_idx = i
            best_mask = comp

    return best_idx, best_mask


def build_full_table_mask_from_kmeans(img):
    h, w = img.shape[:2]

    img_small, labels_2d, _ = run_kmeans_labels(img, k=5, target_w=360)
    cluster_components = build_cluster_component_masks(labels_2d)

    cloth_idx, cloth_small = select_cloth_component(img_small, cluster_components)
    if cloth_small is None:
        return None, None

    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    cloth_dil = cv2.dilate(cloth_small, kernel_dil, iterations=1)
    cloth_er = cv2.erode(cloth_small, kernel_erode, iterations=1)
    ring = cv2.subtract(cloth_dil, cloth_er)

    border_ring = np.zeros_like(cloth_small)
    k = int(labels_2d.max()) + 1

    for i in range(k):
        if i == cloth_idx:
            continue

        cluster_mask = np.zeros_like(cloth_small)
        cluster_mask[labels_2d == i] = 255
        cluster_in_ring = cv2.bitwise_and(cluster_mask, ring)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cluster_in_ring = cv2.morphologyEx(cluster_in_ring, cv2.MORPH_OPEN, kernel_open, iterations=1)
        cluster_in_ring = cv2.morphologyEx(cluster_in_ring, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        if np.sum(cluster_in_ring > 0) < 20:
            continue

        border_ring = cv2.bitwise_or(border_ring, cluster_in_ring)

    merged_small = cv2.bitwise_or(cloth_small, border_ring)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 7))
    merged_small = cv2.morphologyEx(merged_small, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    merged_small = component_containing_seed(merged_small, cloth_small)
    if merged_small is None:
        merged_small = cloth_small.copy()

    flood = merged_small.copy()
    flood_h, flood_w = flood.shape[:2]
    flood_mask = np.zeros((flood_h + 2, flood_w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled_small = cv2.bitwise_or(merged_small, flood_inv)

    filled_small = component_containing_seed(filled_small, cloth_small)
    if filled_small is None:
        filled_small = merged_small.copy()

    full_table_mask = cv2.resize(filled_small, (w, h), interpolation=cv2.INTER_NEAREST)
    cloth_mask_full = cv2.resize(cloth_small, (w, h), interpolation=cv2.INTER_NEAREST)

    kernel_close_full = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    kernel_open_full = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    full_table_mask = cv2.morphologyEx(full_table_mask, cv2.MORPH_CLOSE, kernel_close_full, iterations=2)
    full_table_mask = cv2.morphologyEx(full_table_mask, cv2.MORPH_OPEN, kernel_open_full, iterations=1)

    cloth_mask_full = cv2.morphologyEx(cloth_mask_full, cv2.MORPH_CLOSE, kernel_close_full, iterations=1)
    cloth_mask_full = cv2.morphologyEx(cloth_mask_full, cv2.MORPH_OPEN, kernel_open_full, iterations=1)

    full_table_mask = component_containing_seed(full_table_mask, cloth_mask_full)
    if full_table_mask is None:
        return None, None

    return full_table_mask, cloth_mask_full


def build_edge_maps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)

    canny_gray = cv2.Canny(gray_blur, 50, 150)
    canny_v = cv2.Canny(v_blur, 50, 150)
    edge_union = cv2.bitwise_or(canny_gray, canny_v)

    sobelx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx * sobelx + sobely * sobely)

    return {
        "gray": gray_blur,
        "v": v_blur,
        "edge_union": edge_union,
        "grad_mag": grad_mag,
    }


def refine_quad_corners_with_goodFeaturesToTrack(img, rough_quad, fitted_lines=None):
    if img is None or rough_quad is None:
        return rough_quad, np.zeros(4, dtype=np.float32)

    q = np.asarray(rough_quad, dtype=np.float32).copy()
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    diag = max(np.linalg.norm(q[2] - q[0]), np.linalg.norm(q[3] - q[1]), 1.0)
    roi_r = int(round(0.06 * diag))
    roi_r = max(18, min(roi_r, 80))

    refined = []
    local_scores = []

    for i in range(4):
        cx, cy = q[i]
        x0 = max(0, int(round(cx)) - roi_r)
        y0 = max(0, int(round(cy)) - roi_r)
        x1 = min(w, int(round(cx)) + roi_r + 1)
        y1 = min(h, int(round(cy)) + roi_r + 1)

        roi = gray[y0:y1, x0:x1]
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            refined.append(q[i].copy())
            local_scores.append(0.0)
            continue

        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=25,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=5,
            useHarrisDetector=False
        )

        if pts is None:
            refined.append(q[i].copy())
            local_scores.append(0.0)
            continue

        pts = pts.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += x0
        pts[:, 1] += y0

        best_pt = q[i].copy()
        best_score = -1e18
        num_valid = 0

        for p in pts:
            d_corner = np.linalg.norm(p - q[i])
            if d_corner > roi_r:
                continue

            score = -1.0 * d_corner
            is_valid = True

            if fitted_lines is not None and len(fitted_lines) == 4:
                if i == 0:
                    l1, l2 = fitted_lines[0], fitted_lines[3]
                elif i == 1:
                    l1, l2 = fitted_lines[0], fitted_lines[1]
                elif i == 2:
                    l1, l2 = fitted_lines[1], fitted_lines[2]
                else:
                    l1, l2 = fitted_lines[2], fitted_lines[3]

                d1 = point_line_distance(p, l1)
                d2 = point_line_distance(p, l2)

                score += -2.5 * (d1 + d2)

                if d1 > 20 or d2 > 20:
                    is_valid = False

            if not is_valid:
                continue

            num_valid += 1

            if score > best_score:
                best_score = score
                best_pt = p.copy()

        refined.append(best_pt)

        drift = float(np.linalg.norm(best_pt - q[i]))
        density_score = np.clip(num_valid / 8.0, 0.0, 1.0)
        drift_score = np.clip(1.0 - drift / 15.0, 0.0, 1.0)
        local_score = 0.6 * density_score + 0.4 * drift_score
        local_scores.append(float(np.clip(local_score, 0.0, 1.0)))

    refined = np.asarray(refined, dtype=np.float32)
    local_scores = np.asarray(local_scores, dtype=np.float32)

    try:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)
        pts_sub = refined.reshape(-1, 1, 2).copy()
        cv2.cornerSubPix(gray, pts_sub, (5, 5), (-1, -1), term)
        refined = pts_sub.reshape(-1, 2)
    except Exception:
        pass

    refined = order_quad_tl_tr_br_bl(refined)

    if polygon_area(refined) < 1000:
        return q, np.zeros(4, dtype=np.float32)

    return refined, local_scores


def compute_line_residuals(points, line):
    if points is None or len(points) == 0 or line is None:
        return None
    pts = np.asarray(points, dtype=np.float32)
    a, b, c = line
    return np.abs(a * pts[:, 0] + b * pts[:, 1] + c)


def side_confidence_from_support(points, line, base_line=None, source_kind="unknown"):
    if line is None or points is None or len(points) < 4:
        return 0.0

    pts = np.asarray(points, dtype=np.float32)
    resid = compute_line_residuals(pts, line)
    if resid is None or len(resid) == 0:
        return 0.0

    n_pts = len(pts)
    med_resid = float(np.median(resid))

    angle_penalty = 0.0
    if base_line is not None:
        angle_penalty = np.rad2deg(angle_diff_pi(line_angle(line), line_angle(base_line)))

    qty_score = np.clip((n_pts - 6) / 30.0, 0.0, 1.0)
    fit_score = np.clip(1.0 - med_resid / 4.0, 0.0, 1.0)
    ang_score = np.clip(1.0 - angle_penalty / 12.0, 0.0, 1.0)

    if source_kind == "hull":
        src_score = 1.0
    elif source_kind == "contour":
        src_score = 0.85
    elif source_kind == "rail":
        src_score = 0.70
    elif source_kind == "base":
        src_score = 0.45
    else:
        src_score = 0.60

    conf = 0.35 * qty_score + 0.35 * fit_score + 0.20 * ang_score + 0.10 * src_score
    return float(np.clip(conf, 0.0, 1.0))


def adjacent_side_ids_for_corner(corner_idx):
    if corner_idx == 0:
        return 0, 3
    if corner_idx == 1:
        return 0, 1
    if corner_idx == 2:
        return 1, 2
    return 2, 3


def sample_edge_points_along_parallel_strip(
    edge_maps,
    ref_side_p1,
    ref_side_p2,
    quad_center,
    inward_width=6,
    outward_width=70,
    t0=0.08,
    t1=0.92,
    samples=140,
    step=1,
):
    p1 = np.asarray(ref_side_p1, dtype=np.float32)
    p2 = np.asarray(ref_side_p2, dtype=np.float32)
    center = np.asarray(quad_center, dtype=np.float32)

    n = line_side_normal(p1, p2)
    mid = 0.5 * (p1 + p2)

    if np.dot(center - mid, n) > 0:
        n = -n

    edge_union = edge_maps["edge_union"]
    grad_mag = edge_maps["grad_mag"]

    h, w = edge_union.shape[:2]
    grad_thresh = max(18.0, float(np.percentile(grad_mag, 82)))

    pts = []
    for t in np.linspace(t0, t1, samples):
        base = (1.0 - t) * p1 + t * p2

        best_pt = None
        best_score = -1e18

        for d in range(-inward_width, outward_width + 1, step):
            pt = base + d * n
            x = int(round(pt[0]))
            y = int(round(pt[1]))
            if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
                continue

            e = 1.0 if edge_union[y, x] > 0 else 0.0
            g = float(grad_mag[y, x])

            score = 2.0 * e + 0.03 * g
            if score > best_score:
                best_score = score
                best_pt = np.array([x, y], dtype=np.float32)

            if e > 0.0 and g >= grad_thresh:
                best_pt = np.array([x, y], dtype=np.float32)
                break

        if best_pt is not None and best_score >= 1.2:
            pts.append(best_pt)

    if len(pts) < 8:
        return None

    return np.asarray(pts, dtype=np.float32)


def fit_line_from_edge_strip(
    edge_maps,
    side_p1,
    side_p2,
    quad_center,
    inward_width=5,
    outward_width=70,
    samples=150,
    step=1,
    angle_ref_line=None,
    max_angle_deg=12.0,
    inlier_thresh=3.5,
):
    pts = sample_edge_points_along_parallel_strip(
        edge_maps,
        side_p1,
        side_p2,
        quad_center,
        inward_width=inward_width,
        outward_width=outward_width,
        t0=0.08,
        t1=0.92,
        samples=samples,
        step=step,
    )

    if pts is None or len(pts) < 8:
        return None, None

    line = robust_fit_line(pts, max_iter=8, inlier_thresh=inlier_thresh)
    if line is None:
        return None, pts

    if angle_ref_line is not None:
        ang_err = np.rad2deg(angle_diff_pi(line_angle(line), line_angle(angle_ref_line)))
        if ang_err > max_angle_deg:
            return None, pts

    return line, pts


def fit_line_from_support_points(pts, base_line, max_angle_deg=12.0, inlier_thresh=3.0):
    if pts is None or len(pts) < 8:
        return None

    line = robust_fit_line(pts, max_iter=8, inlier_thresh=inlier_thresh)
    if line is None:
        return None

    ang_err = np.rad2deg(angle_diff_pi(line_angle(line), line_angle(base_line)))
    if ang_err > max_angle_deg:
        return None

    return line


def score_side_candidate(points, line, base_line, source_kind):
    if line is None or points is None or len(points) < 4:
        return -1e18

    resid = compute_line_residuals(points, line)
    if resid is None or len(resid) == 0:
        return -1e18

    med_resid = float(np.median(resid))
    ang_err = np.rad2deg(angle_diff_pi(line_angle(line), line_angle(base_line)))
    n_pts = len(points)

    if source_kind == "hull":
        src_bonus = 10.0
    elif source_kind == "contour":
        src_bonus = 7.0
    elif source_kind == "rail":
        src_bonus = 6.0
    else:
        src_bonus = 0.0

    return 1.0 * n_pts - 2.5 * med_resid - 2.0 * ang_err + src_bonus


def signed_dist_points_to_line(points, line):
    pts = np.asarray(points, dtype=np.float32)
    a, b, c = line
    return a * pts[:, 0] + b * pts[:, 1] + c


def reanchor_parallel_line_to_points(line, anchor_points, fallback_point=None, max_shift=40.0):
    if line is None:
        return None

    a, b, c = line

    if anchor_points is not None and len(anchor_points) >= 3:
        vals = signed_dist_points_to_line(anchor_points, line)
        delta = float(np.median(vals))
        delta = float(np.clip(delta, -max_shift, max_shift))
        return np.array([a, b, c - delta], dtype=np.float32)

    if fallback_point is not None:
        x, y = float(fallback_point[0]), float(fallback_point[1])
        c_new = -(a * x + b * y)
        return np.array([a, b, c_new], dtype=np.float32)

    return line


def collect_anchor_points_for_side(contour_pts, hull_pts, p1, p2):
    pts_hull = collect_points_near_segment_trimmed(
        hull_pts, p1, p2, dist_thresh=8.0, ext_frac=0.08, trim_t=0.18
    )

    pts_cont = collect_points_near_segment_trimmed(
        contour_pts, p1, p2, dist_thresh=8.0, ext_frac=0.06, trim_t=0.24
    )

    chunks = []
    if pts_hull is not None and len(pts_hull) >= 3:
        chunks.append(pts_hull)
    if pts_cont is not None and len(pts_cont) >= 3:
        chunks.append(pts_cont)

    if len(chunks) == 0:
        return None

    return np.vstack(chunks).astype(np.float32)


def collect_endpoint_anchor_points_for_side(contour_pts, hull_pts, p1, p2, radius=55):
    chunks = []

    for src in [contour_pts, hull_pts]:
        a = collect_points_in_disk(src, p1, radius=radius)
        b = collect_points_in_disk(src, p2, radius=radius)
        if a is not None and len(a) >= 3:
            chunks.append(a)
        if b is not None and len(b) >= 3:
            chunks.append(b)

    if len(chunks) == 0:
        return None

    return np.vstack(chunks).astype(np.float32)


def fit_side_line_generic(
    idx,
    rough_sides,
    contour_pts,
    hull_pts,
    edge_maps,
    quad_center,
    allow_contour=True,
    allow_hull=True,
    allow_rail=True,
):
    p1, p2 = rough_sides[idx]
    base_line = line_from_two_points(p1, p2)
    side_mid = 0.5 * (np.asarray(p1, np.float32) + np.asarray(p2, np.float32))

    pts_hull = None
    pts_cont = None
    rail_pts = None

    if allow_hull:
        pts_hull = collect_points_near_segment_trimmed(
            hull_pts, p1, p2, dist_thresh=7.5, ext_frac=0.08, trim_t=0.18
        )

    if allow_contour:
        pts_cont = collect_points_near_segment_trimmed(
            contour_pts, p1, p2, dist_thresh=7.5, ext_frac=0.06, trim_t=0.22
        )

    rail_line = None
    if allow_rail and edge_maps is not None:
        rail_line, rail_pts = fit_line_from_edge_strip(
            edge_maps,
            p1,
            p2,
            quad_center,
            inward_width=5,
            outward_width=85,
            samples=160,
            step=1,
            angle_ref_line=base_line,
            max_angle_deg=14.0,
            inlier_thresh=3.5,
        )

        if rail_line is not None:
            anchor_pts_local = collect_endpoint_anchor_points_for_side(
                contour_pts, hull_pts, p1, p2, radius=60
            )
            anchor_pts_global = collect_anchor_points_for_side(contour_pts, hull_pts, p1, p2)
            anchor_pts = anchor_pts_local if anchor_pts_local is not None else anchor_pts_global

            rail_line = reanchor_parallel_line_to_points(
                rail_line,
                anchor_points=anchor_pts,
                fallback_point=side_mid,
                max_shift=28.0,
            )

    candidates = []

    hull_line = fit_line_from_support_points(pts_hull, base_line, max_angle_deg=10.0, inlier_thresh=2.8)
    if hull_line is not None:
        candidates.append(("hull", hull_line, pts_hull))

    cont_line = fit_line_from_support_points(pts_cont, base_line, max_angle_deg=10.0, inlier_thresh=2.8)
    if cont_line is not None:
        candidates.append(("contour", cont_line, pts_cont))

    if rail_line is not None and rail_pts is not None and len(rail_pts) >= 8:
        candidates.append(("rail", rail_line, rail_pts))

    if len(candidates) == 0:
        return base_line, None, "base", 0.35

    best = None
    best_score = -1e18
    for src, line, pts in candidates:
        sc = score_side_candidate(pts, line, base_line, src)
        if src == "rail":
            sc -= 7.0
        if sc > best_score:
            best_score = sc
            best = (src, line, pts)

    src, line, pts = best
    conf = side_confidence_from_support(pts, line, base_line=base_line, source_kind=src)

    if src == "rail":
        conf = min(conf, 0.65)

    return line, pts, src, conf


def detect_quad_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1000:
        return None

    hull = cv2.convexHull(contour, clockwise=False)
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    if len(hull_pts) < 12:
        return None

    h, w = mask.shape[:2]

    hull_edge = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(hull_edge, [hull.astype(np.int32)], True, 255, 2)

    lines_p = cv2.HoughLinesP(
        hull_edge,
        rho=1,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=max(30, int(min(w, h) * 0.18)),
        maxLineGap=20,
    )

    if lines_p is None or len(lines_p) < 4:
        return None

    segments = []
    for l in lines_p[:, 0, :]:
        x1, y1, x2, y2 = map(float, l)
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        if length < 20:
            continue

        angle = np.arctan2(dy, dx)
        if angle < 0:
            angle += np.pi

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        segments.append({
            "p1": np.array([x1, y1], dtype=np.float32),
            "p2": np.array([x2, y2], dtype=np.float32),
            "len": float(length),
            "angle": float(angle),
            "mx": float(mx),
            "my": float(my),
        })

    if len(segments) < 4:
        return None

    def angle_dist_to_horizontal(a):
        return min(abs(a - 0.0), abs(a - np.pi))

    horiz = []
    vert = []
    for s in segments:
        if angle_dist_to_horizontal(s["angle"]) < np.deg2rad(30):
            horiz.append(s)
        else:
            vert.append(s)

    if len(horiz) < 2 or len(vert) < 2:
        return None

    def take_side_group(cands, key_name, target="small", n_take=6):
        vals = np.array([c[key_name] for c in cands], dtype=np.float32)
        if target == "small":
            pivot = np.quantile(vals, 0.35)
            sel = [c for c in cands if c[key_name] <= pivot]
        else:
            pivot = np.quantile(vals, 0.65)
            sel = [c for c in cands if c[key_name] >= pivot]

        sel = sorted(sel, key=lambda s: s["len"], reverse=True)
        return sel[:n_take]

    top_group = take_side_group(horiz, "my", target="small", n_take=6)
    bottom_group = take_side_group(horiz, "my", target="large", n_take=6)
    left_group = take_side_group(vert, "mx", target="small", n_take=6)
    right_group = take_side_group(vert, "mx", target="large", n_take=6)

    if min(len(top_group), len(bottom_group), len(left_group), len(right_group)) < 1:
        return None

    def segments_to_points(group):
        pts = []
        for s in group:
            pts.append(s["p1"])
            pts.append(s["p2"])
        return np.asarray(pts, dtype=np.float32)

    top_line = robust_fit_line(segments_to_points(top_group), max_iter=4, inlier_thresh=2.5)
    bottom_line = robust_fit_line(segments_to_points(bottom_group), max_iter=4, inlier_thresh=2.5)
    left_line = robust_fit_line(segments_to_points(left_group), max_iter=4, inlier_thresh=2.5)
    right_line = robust_fit_line(segments_to_points(right_group), max_iter=4, inlier_thresh=2.5)

    if any(line is None for line in [top_line, bottom_line, left_line, right_line]):
        return None

    tl = intersect_lines(top_line, left_line)
    tr = intersect_lines(top_line, right_line)
    br = intersect_lines(bottom_line, right_line)
    bl = intersect_lines(bottom_line, left_line)

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    quad = order_quad_tl_tr_br_bl(quad)

    if polygon_area(quad) < 1000:
        return None

    return quad


def detect_quad_from_rail_strips(img, full_table_mask, cloth_mask_full):
    if cloth_mask_full is None:
        return None

    contours, _ = cv2.findContours(cloth_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1000:
        return None

    contour_pts = contour.reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(contour)
    hull_pts = hull.reshape(-1, 2).astype(np.float32)

    peri = cv2.arcLength(hull, True)
    rough_quad = None

    for frac in np.linspace(0.01, 0.08, 28):
        approx = cv2.approxPolyDP(hull, frac * peri, True)
        if len(approx) == 4:
            q = approx.reshape(-1, 2).astype(np.float32)
            q = order_quad_tl_tr_br_bl(q)
            if polygon_area(q) > 1000:
                rough_quad = q
                break

    if rough_quad is None:
        rect = cv2.minAreaRect(contour)
        rough_quad = order_quad_tl_tr_br_bl(cv2.boxPoints(rect).astype(np.float32))

    if rough_quad is None or polygon_area(rough_quad) < 1000:
        return None

    quad_center = np.mean(rough_quad, axis=0)

    rough_sides = [
        (rough_quad[0], rough_quad[1]),
        (rough_quad[1], rough_quad[2]),
        (rough_quad[2], rough_quad[3]),
        (rough_quad[3], rough_quad[0]),
    ]

    fitted = [None, None, None, None]
    side_conf = [0.0, 0.0, 0.0, 0.0]
    side_support_used = [None, None, None, None]
    side_source_kind = ["unknown"] * 4

    edge_maps = build_edge_maps(img) if img is not None else None

    for idx in range(4):
        line, pts, src, conf = fit_side_line_generic(
            idx=idx,
            rough_sides=rough_sides,
            contour_pts=contour_pts,
            hull_pts=hull_pts,
            edge_maps=edge_maps,
            quad_center=quad_center,
            allow_contour=True,
            allow_hull=True,
            allow_rail=True,
        )

        fitted[idx] = line
        side_support_used[idx] = pts
        side_source_kind[idx] = src
        side_conf[idx] = conf

    opposite_pairs = [(0, 2), (1, 3)]

    for a, b in opposite_pairs:
        if side_conf[a] >= 0.55 and side_conf[b] < 0.55:
            cand_line, cand_pts, cand_src, cand_conf = fit_side_line_generic(
                idx=b,
                rough_sides=rough_sides,
                contour_pts=contour_pts,
                hull_pts=hull_pts,
                edge_maps=edge_maps,
                quad_center=quad_center,
                allow_contour=False,
                allow_hull=True,
                allow_rail=True,
            )
            if cand_conf > side_conf[b]:
                fitted[b] = cand_line
                side_support_used[b] = cand_pts
                side_source_kind[b] = cand_src
                side_conf[b] = cand_conf

        if side_conf[b] >= 0.55 and side_conf[a] < 0.55:
            cand_line, cand_pts, cand_src, cand_conf = fit_side_line_generic(
                idx=a,
                rough_sides=rough_sides,
                contour_pts=contour_pts,
                hull_pts=hull_pts,
                edge_maps=edge_maps,
                quad_center=quad_center,
                allow_contour=False,
                allow_hull=True,
                allow_rail=True,
            )
            if cand_conf > side_conf[a]:
                fitted[a] = cand_line
                side_support_used[a] = cand_pts
                side_source_kind[a] = cand_src
                side_conf[a] = cand_conf

    for a, b in opposite_pairs:
        if side_source_kind[a] == "base" and side_conf[b] >= 0.60 and edge_maps is not None:
            p1, p2 = rough_sides[a]
            base_line = line_from_two_points(p1, p2)
            rail_line, rail_pts = fit_line_from_edge_strip(
                edge_maps=edge_maps,
                side_p1=p1,
                side_p2=p2,
                quad_center=quad_center,
                inward_width=5,
                outward_width=95,
                samples=180,
                step=1,
                angle_ref_line=base_line,
                max_angle_deg=14.0,
                inlier_thresh=3.5,
            )
            if rail_line is not None and rail_pts is not None and len(rail_pts) >= 8:
                fitted[a] = rail_line
                side_support_used[a] = rail_pts
                side_source_kind[a] = "rail"
                side_conf[a] = side_confidence_from_support(
                    rail_pts, rail_line, base_line=base_line, source_kind="rail"
                )

        if side_source_kind[b] == "base" and side_conf[a] >= 0.60 and edge_maps is not None:
            p1, p2 = rough_sides[b]
            base_line = line_from_two_points(p1, p2)
            rail_line, rail_pts = fit_line_from_edge_strip(
                edge_maps=edge_maps,
                side_p1=p1,
                side_p2=p2,
                quad_center=quad_center,
                inward_width=5,
                outward_width=95,
                samples=180,
                step=1,
                angle_ref_line=base_line,
                max_angle_deg=14.0,
                inlier_thresh=3.5,
            )
            if rail_line is not None and rail_pts is not None and len(rail_pts) >= 8:
                fitted[b] = rail_line
                side_support_used[b] = rail_pts
                side_source_kind[b] = "rail"
                side_conf[b] = side_confidence_from_support(
                    rail_pts, rail_line, base_line=base_line, source_kind="rail"
                )

    weak_pairs = 0
    for ci in range(4):
        s1, s2 = adjacent_side_ids_for_corner(ci)
        if side_conf[s1] < 0.50 and side_conf[s2] < 0.50:
            weak_pairs += 1

    if weak_pairs >= 2:
        return None

    tl = intersect_lines(fitted[0], fitted[3])
    tr = intersect_lines(fitted[0], fitted[1])
    br = intersect_lines(fitted[1], fitted[2])
    bl = intersect_lines(fitted[2], fitted[3])

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    quad = order_quad_tl_tr_br_bl(quad)

    if polygon_area(quad) < 2000:
        return None

    side_lens_final = side_lengths_of_quad(quad)
    if min(side_lens_final) < 40:
        return None

    quad_refined, corner_local_scores = refine_quad_corners_with_goodFeaturesToTrack(
        img, quad, fitted_lines=fitted
    )

    if quad_refined is not None and polygon_area(quad_refined) > 2000:
        quad_refined = np.asarray(quad_refined, dtype=np.float32)

        for ci in range(4):
            s1, s2 = adjacent_side_ids_for_corner(ci)
            corner_side_conf = min(side_conf[s1], side_conf[s2])
            local_conf = float(corner_local_scores[ci])

            final_corner_conf = 0.65 * corner_side_conf + 0.35 * local_conf
            drift = np.linalg.norm(quad_refined[ci] - quad[ci])

            if final_corner_conf < 0.52 or drift > 12.0:
                quad_refined[ci] = quad[ci]

        quad = quad_refined

    return quad


def quad_to_mask(shape_hw, quad):
    h, w = shape_hw
    q = np.asarray(quad, dtype=np.float32)
    if q.shape != (4, 2):
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 255)
    return mask


def validate_quad_against_mask(
    quad,
    mask,
    min_area=1500,
    min_cover=0.80,
    min_precision=0.35,
    max_outside_frac=0.50,
):
    if quad is None or mask is None:
        return False

    q = np.asarray(quad, dtype=np.float32)
    if q.shape != (4, 2):
        return False
    if not np.all(np.isfinite(q)):
        return False

    if polygon_area(q) < min_area:
        return False

    h, w = mask.shape[:2]

    outside = 0
    for x, y in q:
        if x < -0.25 * w or x > 1.25 * w or y < -0.25 * h or y > 1.25 * h:
            outside += 1
    if outside >= 2:
        return False

    quad_mask = quad_to_mask((h, w), q)
    if quad_mask is None:
        return False

    cloth = mask > 0
    quad_bool = quad_mask > 0

    cloth_area = int(np.sum(cloth))
    quad_area = int(np.sum(quad_bool))
    inter = int(np.sum(cloth & quad_bool))

    if cloth_area < 100 or quad_area < 100:
        return False

    cover = inter / float(cloth_area)
    precision = inter / float(quad_area)
    outside_frac = max(0.0, (quad_area - inter) / float(max(quad_area, 1)))

    if cover < min_cover:
        return False
    if precision < min_precision:
        return False
    if outside_frac > max_outside_frac:
        return False

    return True


def detect_table_corners(img):
    full_table_mask, cloth_mask_full = build_full_table_mask_from_kmeans(img)
    if full_table_mask is None or cloth_mask_full is None:
        return None

    quad = detect_quad_from_rail_strips(img, full_table_mask, cloth_mask_full)

    quad_valid = validate_quad_against_mask(
        quad,
        cloth_mask_full,
        min_area=1500,
        min_cover=0.80,
        min_precision=0.35,
        max_outside_frac=0.50,
    )

    if quad is None or not quad_valid:
        quad = detect_quad_from_mask(cloth_mask_full)

    if quad is None:
        return None

    quad = order_quad_tl_tr_br_bl(quad)

    if polygon_area(quad) < 2000:
        return None

    return quad


def generate_top_view(img, corners, width=FULL_TABLE_WIDTH, height=FULL_TABLE_HEIGHT):
    corners = np.asarray(corners, dtype=np.float32)

    top = np.linalg.norm(corners[1] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])

    if width is None and height is None:
        width = int(round(0.5 * (top + bottom)))
        height = int(round(0.5 * (left + right)))
    elif width is not None and height is None:
        aspect = (0.5 * (top + bottom)) / max(0.5 * (left + right), 1.0)
        height = int(round(width / aspect))
    elif width is None and height is not None:
        aspect = (0.5 * (top + bottom)) / max(0.5 * (left + right), 1.0)
        width = int(round(height * aspect))

    width = max(10, int(width))
    height = max(10, int(height))

    dst_pts = np.array([
        [0, height - 1],
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC)
    return warped, M




# ============================================================
# BALL DETECTION IN ORIGINAL IMAGE
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def estimate_cloth_lab(img, playable_mask):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    ys, xs = np.where(playable_mask > 0)
    if len(xs) < 50:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    vals = lab[ys, xs]
    return np.median(vals, axis=0).astype(np.float32)


def build_playable_mask(cloth_mask_full):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.erode(cloth_mask_full, kernel, iterations=1)


def circularity_of_contour(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri < 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def box_mask_overlap_ratio(box, mask):
    x1, y1, x2, y2 = box
    h, w = mask.shape[:2]

    x1 = max(0, min(w - 1, int(round(x1))))
    x2 = max(0, min(w,     int(round(x2))))
    y1 = max(0, min(h - 1, int(round(y1))))
    y2 = max(0, min(h,     int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    return float(np.mean(roi > 0))


def build_ball_score_map(img, cloth_mask_full, full_table_mask=None, debug_dir=None):
    playable = build_playable_mask(cloth_mask_full)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cloth_lab = estimate_cloth_lab(img, playable)

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    dL = L - cloth_lab[0]
    dA = A - cloth_lab[1]
    dB = B - cloth_lab[2]
    dist_lab = np.sqrt(dL * dL + dA * dA + dB * dB)

    gray_blur = cv2.GaussianBlur(gray, (0, 0), 4.0)
    local_contrast = cv2.absdiff(gray, gray_blur).astype(np.float32)

    grad_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    dist_n = np.clip((dist_lab - 8.0) / 18.0, 0.0, 1.0)
    contrast_n = np.clip((local_contrast - 4.0) / 14.0, 0.0, 1.0)
    grad_n = np.clip((grad_mag - 8.0) / 28.0, 0.0, 1.0)

    white_score = np.clip((V - 150.0) / 70.0, 0.0, 1.0) * np.clip((70.0 - S) / 70.0, 0.0, 1.0)
    black_score = np.clip((90.0 - V) / 90.0, 0.0, 1.0)

    ball_score = (
        0.58 * dist_n +
        0.20 * contrast_n +
        0.10 * grad_n +
        0.12 * white_score +
        0.10 * black_score
    )

    if full_table_mask is not None:
        candidate_region = (full_table_mask > 0)
    else:
        candidate_region = (cloth_mask_full > 0)

    score_vis = np.zeros_like(gray, dtype=np.uint8)
    score_vis[candidate_region] = np.clip(ball_score[candidate_region] * 255.0, 0, 255).astype(np.uint8)

    if debug_dir is not None:
        ensure_dir(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "debug_candidate_region.jpg"), candidate_region.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(debug_dir, "debug_ball_score.jpg"), score_vis)

    return playable, ball_score, candidate_region.astype(np.uint8) * 255


def circle_disk_mask(h, w, cx, cy, r):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def circle_ring_mask(h, w, cx, cy, r_inner, r_outer):
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 >= r_inner * r_inner) & (d2 <= r_outer * r_outer)


def dedupe_circles(circles, dist_thresh=18.0, radius_thresh=3.0):
    if not circles:
        return []

    circles = sorted(circles, key=lambda c: c["score"], reverse=True)
    kept = []

    for c in circles:
        ok = True
        for kc in kept:
            d = np.hypot(c["cx"] - kc["cx"], c["cy"] - kc["cy"])
            dr = abs(c["r"] - kc["r"])
            if d < dist_thresh and dr < radius_thresh:
                ok = False
                break
        if ok:
            kept.append(c)

    return kept



def box_center(box):
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def detect_ball_boxes(img, cloth_mask_full, full_table_mask=None, debug_dir=None):
    h, w = img.shape[:2]

    if debug_dir is not None:
        ensure_dir(debug_dir)

    playable, ball_score, candidate_region = build_ball_score_map(
        img,
        cloth_mask_full,
        full_table_mask=full_table_mask,
        debug_dir=debug_dir
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

    # Use cloth only for spatial validity
    k_center = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    center_region = cv2.erode(cloth_mask_full, k_center, iterations=1)

    masked_gray = gray.copy()
    masked_gray[candidate_region == 0] = 0

    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=90,
        param2=14,
        minRadius=7,
        maxRadius=19
    )

    if debug_dir is not None:
        dbg_raw = img.copy()
        if circles is not None:
            raw = np.round(circles[0]).astype(int)
            for cx, cy, r in raw:
                cv2.circle(dbg_raw, (cx, cy), r, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, "debug_hough_raw.jpg"), dbg_raw)
        cv2.imwrite(os.path.join(debug_dir, "debug_center_region.jpg"), center_region)

    candidates = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        for cx, cy, r in circles:
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                continue

            disk = circle_disk_mask(h, w, cx, cy, r)
            ring = circle_ring_mask(h, w, cx, cy, max(1, int(0.9 * r)), int(1.5 * r))

            if np.sum(disk) < 20:
                continue

            mean_score = float(np.mean(ball_score[disk]))
            mean_ring_score = float(np.mean(ball_score[ring])) if np.any(ring) else 0.0

            cloth_overlap = float(np.mean(cloth_mask_full[disk] > 0))
            table_mask = cloth_mask_full
            table_overlap = float(np.mean(table_mask[disk] > 0))

            score_gain = mean_score - mean_ring_score
            center_ok = int(center_region[cy, cx] > 0)

            if debug_dir is not None:
                if (cx < 800 and cy < 260) or (cx > 1400 and cy > 700):
                    print(
                        f"circle cx={cx} cy={cy} r={r} "
                        f"center_ok={center_ok} "
                        f"mean_score={mean_score:.3f} "
                        f"score_gain={score_gain:.3f} "
                        f"cloth_overlap={cloth_overlap:.3f} "
                        f"table_overlap={table_overlap:.3f}"
                    )

            if center_ok == 0:
                continue

            if mean_score < 0.50:
                continue
            if score_gain < 0.04:
                continue
            if cloth_overlap < 0.18 and table_overlap < 0.30:
                continue

            candidates.append({
                "cx": int(cx),
                "cy": int(cy),
                "r": int(r),
                "score": mean_score
            })

    if debug_dir is not None:
        print("ACCEPTED CANDIDATES BEFORE DEDUPE:")
        for c in candidates:
            print(
                f"cx={c['cx']} cy={c['cy']} r={c['r']} score={c['score']:.3f}"
            )

    candidates = dedupe_circles(candidates, dist_thresh=20.0)

    if debug_dir is not None:
        print("ACCEPTED CANDIDATES AFTER DEDUPE:")
        for c in candidates:
            print(
                f"cx={c['cx']} cy={c['cy']} r={c['r']} score={c['score']:.3f}"
            )

    boxes = []
    for c in candidates:
        cx, cy, r = c["cx"], c["cy"], c["r"]
        x1 = max(0, int(round(cx - r)))
        y1 = max(0, int(round(cy - r)))
        x2 = min(w - 1, int(round(cx + r)))
        y2 = min(h - 1, int(round(cy + r)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    if debug_dir is not None:
        cand_vis = np.zeros((h, w), dtype=np.uint8)
        cand_vis[(ball_score >= 0.16) & (candidate_region > 0)] = 255
        cv2.imwrite(os.path.join(debug_dir, "debug_ball_candidate_mask.jpg"), cand_vis)
        cv2.imwrite(os.path.join(debug_dir, "debug_playable_mask.jpg"), playable)

        dbg = img.copy()
        for c in candidates:
            cv2.circle(dbg, (c["cx"], c["cy"]), c["r"], (0, 255, 255), 1)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "debug_ball_boxes.jpg"), dbg)

    balls = []
    for x1, y1, x2, y2 in boxes:
        balls.append({
            "xmin": float(x1 / w),
            "xmax": float(x2 / w),
            "ymin": float(y1 / h),
            "ymax": float(y2 / h),
        })

    return balls

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    debug_dir = os.path.join("debug", filename_base)
    ensure_dir(debug_dir)

    full_table_mask, cloth_mask_full = build_full_table_mask_from_kmeans(img)
    if full_table_mask is None or cloth_mask_full is None:
        return None, None

    cv2.imwrite(os.path.join(debug_dir, "debug_full_table_mask.jpg"), full_table_mask)
    cv2.imwrite(os.path.join(debug_dir, "debug_cloth_mask_full.jpg"), cloth_mask_full)

    corners = detect_table_corners(img)
    if corners is None:
        return None, None

    top_view, _ = generate_top_view(img, corners)
    cv2.imwrite(os.path.join(debug_dir, "debug_top_view.jpg"), top_view)

    h, w = img.shape[:2]

    balls = detect_ball_boxes(
        img,
        cloth_mask_full,
        full_table_mask=full_table_mask,
        debug_dir=debug_dir
    )

    result = {
        "image_path": image_path,
        "table_corners": [
            {"id": 0, "x": float(corners[0][0] / w), "y": float(corners[0][1] / h)},
            {"id": 1, "x": float(corners[1][0] / w), "y": float(corners[1][1] / h)},
            {"id": 2, "x": float(corners[2][0] / w), "y": float(corners[2][1] / h)},
            {"id": 3, "x": float(corners[3][0] / w), "y": float(corners[3][1] / h)},
        ],
        "num_balls": int(len(balls)),
        "balls": balls,
    }

    return result, top_view

def main():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage:")
        print("  python pool_detector.py input.json")
        sys.exit(1)

    input_path = args[0]

    with open(input_path, "r") as f:
        input_data = json.load(f)

    image_paths = input_data["image_path"]

    output_dir = "top_views"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for img_path in image_paths:
        result, top_view = process_image(img_path)

        if result is not None:
            results.append(result)

        if top_view is not None:
            filename = os.path.basename(img_path)
            top_view_path = os.path.join(output_dir, filename)
            cv2.imwrite(top_view_path, top_view)

    output_path = "output.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")
    print(f"Top-views saved to {output_dir}/")


if __name__ == "__main__":
    main()