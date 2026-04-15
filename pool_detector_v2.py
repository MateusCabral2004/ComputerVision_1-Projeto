import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def show(title, img):
    plt.figure(figsize=(10, 6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not found: "+ path)
    return img


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

def warp_from_cluster(img, mask):
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
        return None

    outer = max(contours, key=cv2.contourArea)
    outer_mask = np.zeros_like(mask)
    cv2.drawContours(outer_mask, [outer], -1, 255, 3)

    padded = cv2.copyMakeBorder(outer_mask, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, 0)

    edges = cv2.Canny(padded, 100, 200)
    #show("Edges", edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=140, maxLineGap=200)

    if lines is None:
        return None

    #Draw the lines on a blank image for visualization
    line_img = np.zeros_like(edges)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
    #show("Detected Lines", line_img)

    shifted = [np.array([[x1 - PAD, y1 - PAD, x2 - PAD, y2 - PAD]]) for x1, y1, x2, y2 in lines[:, 0]]

    merged = merge_lines(shifted)
    merged = sorted(merged,key=lambda l: (np.hypot(l[5]-l[3], l[4]-l[2]), abs(l[0])),reverse=True)[:4]

    if len(merged) < 4:
        return None

    #Draw merged lines for visualization
    line_img = np.zeros_like(edges)
    for m, b, x1, y1, x2, y2 in merged:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
    #show("Merged Lines", line_img)

    corners = compute_corners_from_lines(merged)
    if not corners:
        return None

    src = order_points(corners) + PAD

    dst = np.array([[0, 800], [0, 0], [1600, 0], [1600, 800]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)

    padded_img = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_temp = cv2.warpPerspective(padded_img, M, (1600, 800))
    if np.count_nonzero(img_temp) < (img_temp.size / 2):
        return None

    return img_temp

def extract_blue_cloth(warped_img, padding=50):
    if warped_img is None:
        return None

    # 1. Convert to HSV for robust color detection
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

    # 2. Define Blue range (matching your previous logic)
    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 3. Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Find the largest blue contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No blue cloth detected in warped image.")
        return warped_img # Return original warped if nothing found

    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)

    # 5. Expand the bounding box by 50 pixels
    img_h, img_w = warped_img.shape[:2]

    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    w_new = min(img_w - x_new, w + (2 * padding))
    h_new = min(img_h - y_new, h + (2 * padding))

    # 6. Crop and return
    return warped_img[y_new : y_new + h_new, x_new : x_new + w_new]


def top_view(image_path):
    img = load_image(image_path)

    labels, shape, K = kmeans_cluster_refined(img, 5)
    masks = get_cluster_masks(labels, shape, K)

    blue_idx = find_blue_cluster(img, labels, K)
    target = select_target_cluster(img, labels, masks, blue_idx)

    warped = warp_from_cluster(img, masks[target])

    #show("Original", img)
    #show("Target Mask", masks[target])
    top_view_image = None
    if warped is not None:
        top_view_image = warped
        #show("Warped", warped)
    else:
        img2 = extract_blue_cloth(img, padding=50)
        top_view_image = img2
        #show("Fallback Blue Cloth", img2)

    image_name = os.path.basename(image_path)
    save_image("top_views/" + image_name, top_view_image)

def apply_nms(candidates, overlap_thresh=0.15):
    if not candidates: return []
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    kept = []
    for cand in candidates:
        boxA = cand["box"]
        discard = False
        for k in kept:
            boxB = k["box"]
            xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
            xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            areaA, areaB = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]), (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            denom = min(areaA, areaB)
            if denom > 0 and (interArea / denom) > overlap_thresh:
                discard = True
                break
        if not discard: kept.append(cand)
    return [k["box"] for k in kept]

def save_image(path, img):
    print("Saving:", path)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/top_views", exist_ok=True)
    os.makedirs("results/balls", exist_ok=True)
    cv2.imwrite("results/" + path, img)

def detect_balls(path):
    img = cv2.imread(path)
    if img is None: return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([120, 255, 255])
    table_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel_sm = np.ones((5, 5), np.uint8)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_OPEN, kernel_sm)

    cnts, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img, 0

    valid_cnts = [c for c in cnts if cv2.boundingRect(c)[1] > (h * 0.15)]
    if not valid_cnts: return img, 0

    table_cnt = max(valid_cnts, key=cv2.contourArea)
    refined_mask = np.zeros_like(table_mask)
    cv2.drawContours(refined_mask, [table_cnt], -1, 255, -1)

    safety_mask = cv2.erode(refined_mask, np.ones((28, 28), np.uint8))

    s_inv = cv2.bitwise_not(sat)
    s_masked = cv2.bitwise_and(s_inv, s_inv, mask=safety_mask)

    blurred = cv2.medianBlur(s_masked, 7)
    blurred = cv2.GaussianBlur(blurred, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.1, minDist=35,
        param1=50, param2=15, minRadius=15, maxRadius=24
    )

    candidates = []
    if circles is not None:
        for i, (cx, cy, r) in enumerate(np.round(circles[0]).astype(int)):
            if 0 <= cx < w and 0 <= cy < h and safety_mask[cy, cx] == 255:

                y1, y2, x1, x2 = max(0, cy - r), min(h, cy + r), max(0, cx - r), min(w, cx + r)
                roi_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                roi_hsv = hsv[y1:y2, x1:x2]

                if roi_gray.size > 0:
                    std_dev = np.std(roi_gray)
                    avg_v = np.mean(roi_hsv[:, :, 2])
                    avg_s = np.mean(roi_hsv[:, :, 1])

                    is_dark = avg_v < 80
                    is_saturated = avg_s > 60

                    if std_dev > 3.5 and (is_dark or is_saturated):
                        candidates.append({
                            "box": (cx - r, cy - r, cx + r, cy + r),
                            "score": std_dev * (1.0 + (255 - avg_v) / 255.0)
                        })

    final_boxes = apply_nms(candidates)

    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    image_name = os.path.basename(path)
    save_image("balls/" + image_name, img)

    return img, len(final_boxes)

def process_image(image_path):
    top_view(image_path)
    detect_balls(image_path)

def main():

    development_dir = r"C:\Users\mateu\PycharmProjects\ComputerVision_1-Projeto\development_set"
    for image in os.listdir(development_dir):
        if image.endswith(".jpg"):
            path = os.path.join(development_dir, image)
            process_image(path)


if __name__ == "__main__":
    main()