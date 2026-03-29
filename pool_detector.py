"""
Computer Vision Project - Task 1: 8-Ball Pool Table Analysis
M.IA | FEUP | 2025/2026

Pipeline:
    1. Table detection (find 4 corners of playing surface)
    2. Ball detection (locate balls, compute bounding boxes)
    3. Ball classification (assign number 0-15 based on color)
    4. Top-view generation (perspective warp using table corners)

Usage:
    python pool_detector.py input.json
"""

import cv2
import numpy as np
import json
import sys
import os

# ============================================================
# CONSTANTS
# ============================================================

# Standard pool table aspect ratio (interior playing surface)
# 8-ball tables are typically 9ft x 4.5ft → ratio 2:1
TABLE_WIDTH = 900
TABLE_HEIGHT = 450

# HSV ranges for the blue table cloth (Predator tournament tables)
# These may need tuning based on the dataset
TABLE_BLUE_LOWER = np.array([90, 80, 80])
TABLE_BLUE_UPPER = np.array([120, 255, 255])

# Ball color definitions in HSV space
# Format: (name, number_solid, number_stripe, H_low, H_high, S_low, S_high, V_low, V_high)
# We'll refine these as we test
BALL_COLORS = {
    "yellow":  {"solid": 1,  "stripe": 9,  "h_range": (20, 35),   "s_range": (100, 255), "v_range": (150, 255)},
    "blue":    {"solid": 2,  "stripe": 10, "h_range": (100, 130), "s_range": (100, 255), "v_range": (50, 200)},
    "red":     {"solid": 3,  "stripe": 11, "h_range": (0, 10),    "s_range": (100, 255), "v_range": (100, 255)},
    "purple":  {"solid": 4,  "stripe": 12, "h_range": (130, 160), "s_range": (40, 255),  "v_range": (30, 200)},
    "orange":  {"solid": 5,  "stripe": 13, "h_range": (10, 20),   "s_range": (100, 255), "v_range": (150, 255)},
    "green":   {"solid": 6,  "stripe": 14, "h_range": (35, 85),   "s_range": (80, 255),  "v_range": (50, 200)},
    "maroon":  {"solid": 7,  "stripe": 15, "h_range": (0, 10),    "s_range": (50, 150),  "v_range": (30, 130)},
    "black":   {"solid": 8,  "stripe": 8,  "h_range": (0, 180),   "s_range": (0, 80),    "v_range": (0, 60)},
    "white":   {"solid": 0,  "stripe": 0,  "h_range": (0, 180),   "s_range": (0, 40),    "v_range": (180, 255)},
}


# ============================================================
# 1. TABLE DETECTION
# ============================================================

def detect_table_corners(img):
    """
    Detects the 4 corners of the pool table playing surface.
    
    Strategy:
        1. Convert to HSV and segment the blue cloth
        2. Clean up the mask with morphological operations
        3. Find the largest contour (the table surface)
        4. Approximate to a quadrilateral
        5. Order the 4 corners: [top-left, top-right, bottom-right, bottom-left]
    
    Args:
        img: BGR image (numpy array)
    
    Returns:
        corners: numpy array of shape (4, 2) with ordered corners,
                 or None if detection fails
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Segment the blue table cloth
    mask = cv2.inRange(hsv, TABLE_BLUE_LOWER, TABLE_BLUE_UPPER)
    
    # Morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[WARN] No contours found for table detection")
        return None
    
    # Get the largest contour (should be the table)
    largest = max(contours, key=cv2.contourArea)
    
    # Check if contour is large enough (at least 10% of image area)
    img_area = img.shape[0] * img.shape[1]
    if cv2.contourArea(largest) < 0.10 * img_area:
        print("[WARN] Largest contour too small to be a table")
        return None
    
    # Approximate to polygon
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    # If we got exactly 4 points, great
    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        # Fallback: use the minimum area rectangle
        rect = cv2.minAreaRect(largest)
        corners = cv2.boxPoints(rect).astype(np.float32)
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_corners(corners)
    
    return corners


def order_corners(pts):
    """
    Orders 4 points as: [top-left, top-right, bottom-right, bottom-left].
    
    Uses the sum and difference of coordinates:
    - Top-left has smallest sum (x+y)
    - Bottom-right has largest sum (x+y)
    - Top-right has smallest difference (y-x)
    - Bottom-left has largest difference (y-x)
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)       # x + y
    d = np.diff(pts, axis=1)  # y - x
    
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    
    return ordered


# ============================================================
# 2. TOP-VIEW GENERATION (Homography)
# ============================================================

def generate_top_view(img, corners, width=TABLE_WIDTH, height=TABLE_HEIGHT):
    """
    Applies perspective transform to get a top-down view of the table.
    
    Args:
        img: original BGR image
        corners: 4 ordered corners of the table [TL, TR, BR, BL]
        width: output width in pixels
        height: output height in pixels
    
    Returns:
        warped: the top-view image
        M: the perspective transformation matrix (useful for mapping ball positions)
    """
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped, M


# ============================================================
# 3. BALL DETECTION
# ============================================================

def detect_balls(img, corners):
    """
    Detects pool balls on the table.
    
    Key insight: table cloth is consistently BRIGHT blue (V > ~160).
    All balls are either: different hue, darker, less saturated, or very dark.
    By defining cloth precisely, we catch ALL ball types including blue and black.
    
    Args:
        img: BGR image
        corners: 4 ordered corners of the table
    
    Returns:
        balls: list of dicts with 'center', 'radius', 'bbox', 'contour'
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- Create table interior mask ---
    table_mask = np.zeros((h, w), dtype=np.uint8)
    table_poly = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(table_mask, [table_poly], 255)
    
    # --- Determine adaptive V threshold for cloth ---
    # Sample blue-ish pixels inside the table to find cloth brightness
    blue_hue_mask = cv2.inRange(hsv, np.array([85, 60, 0]), np.array([120, 255, 255]))
    blue_in_table = cv2.bitwise_and(blue_hue_mask, table_mask)
    blue_pixels_v = hsv[:, :, 2][blue_in_table > 0]
    
    if len(blue_pixels_v) > 100:
        # Use 60th percentile of V as the cloth brightness reference
        cloth_v_threshold = int(np.percentile(blue_pixels_v, 60)) - 25
        cloth_v_threshold = max(cloth_v_threshold, 100)  # never go below 100
    else:
        cloth_v_threshold = 160
    
    # --- Define "cloth" mask: bright blue pixels ---
    cloth_mask = cv2.inRange(hsv,
        np.array([85, 50, cloth_v_threshold]),
        np.array([120, 255, 255])
    )
    
    # Clean cloth mask (fill small holes that might be ball shadows)
    cloth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, cloth_kernel, iterations=2)
    
    # --- "Not cloth" = inside table AND not cloth ---
    not_cloth = cv2.bitwise_and(cv2.bitwise_not(cloth_mask), table_mask)
    
    # --- Erode the table edge from the mask to remove rail/cushion artifacts ---
    # Create a ring mask for the edge region and subtract it
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    table_eroded = cv2.erode(table_mask, erode_kernel, iterations=1)
    edge_region = cv2.subtract(table_mask, table_eroded)
    
    # Remove edge artifacts, but keep ball-sized blobs in the edge region
    edge_not_cloth = cv2.bitwise_and(not_cloth, edge_region)
    # Only keep edge blobs if they're small enough to be balls
    edge_contours, _ = cv2.findContours(edge_not_cloth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_area = cv2.contourArea(table_poly)
    max_edge_blob = table_area * 0.008
    for cnt in edge_contours:
        if cv2.contourArea(cnt) > max_edge_blob:
            cv2.drawContours(not_cloth, [cnt], -1, 0, -1)
    
    # --- Morphological cleanup ---
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove noise
    not_cloth = cv2.morphologyEx(not_cloth, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Fill gaps within balls (number markings, highlights, stripes)
    not_cloth = cv2.morphologyEx(not_cloth, cv2.MORPH_CLOSE, kernel_med, iterations=2)
    
    # --- Find contours and filter ---
    contours, _ = cv2.findContours(not_cloth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- Estimate expected ball size from table dimensions ---
    # Pool balls appear as ~4% of average table width in the image
    # (larger than the mathematical 2.25% because balls are 3D spheres
    # and perspective makes near objects appear larger)
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    avg_width = (top_width + bottom_width) / 2
    expected_ball_radius = avg_width * 0.04 / 2
    
    # Size bounds based on expected radius
    min_radius = max(expected_ball_radius * 0.55, 10)
    max_radius = expected_ball_radius * 2.5
    min_ball_area = np.pi * min_radius * min_radius * 0.5
    max_ball_area = np.pi * max_radius * max_radius * 1.5
    # Area threshold for "this might be multiple merged balls"
    merge_area_threshold = np.pi * (expected_ball_radius * 1.6) ** 2
    
    balls = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area < min_ball_area:
            continue
        
        # If contour is too large, try to split it into individual balls
        if area > merge_area_threshold:
            split_balls = _split_merged_contour(not_cloth, cnt, expected_ball_radius, 
                                                 min_ball_area, w, h)
            balls.extend(split_balls)
            continue
        
        if area > max_ball_area:
            continue
        
        # Circularity check
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue
        
        # Get enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)
        
        # Radius bounds
        if radius < min_radius or radius > max_radius:
            continue
        
        # Fill ratio check
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0
        if fill_ratio < 0.3:
            continue
        
        # Bounding box
        xmin = max(0, cx - radius)
        ymin = max(0, cy - radius)
        xmax = min(w, cx + radius)
        ymax = min(h, cy + radius)
        
        balls.append({
            'center': (cx, cy),
            'radius': radius,
            'bbox': (xmin, ymin, xmax, ymax),
            'contour': cnt
        })
    
    # --- Remove detections near pockets ---
    # Pockets are at the 4 corners + 2 mid-points of long sides
    mid_top = (corners[0] + corners[1]) / 2
    mid_bottom = (corners[2] + corners[3]) / 2
    pocket_positions = [corners[0], corners[1], corners[2], corners[3], mid_top, mid_bottom]
    pocket_exclusion_radius = expected_ball_radius * 3.5
    
    balls_filtered = []
    for ball in balls:
        cx, cy = ball['center']
        near_pocket = False
        for pocket in pocket_positions:
            dist = np.sqrt((cx - pocket[0])**2 + (cy - pocket[1])**2)
            if dist < pocket_exclusion_radius:
                near_pocket = True
                break
        if not near_pocket:
            balls_filtered.append(ball)
    balls = balls_filtered
    
    # --- Remove duplicates (balls too close together) ---
    balls = _remove_duplicate_balls(balls, expected_ball_radius)
    
    return balls


def _split_merged_contour(mask, contour, expected_radius, min_area, img_w, img_h):
    """
    Splits a large contour that likely contains multiple merged balls
    using distance transform and local maxima detection.
    """
    # Create isolated mask for this contour
    contour_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    
    # Distance transform
    dist = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
    
    # Find local maxima (ball centers)
    # Threshold at ~60% of expected radius
    thresh_val = expected_radius * 0.5
    _, peaks = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    
    # Find connected components of peaks (each = one ball center)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks)
    
    balls = []
    for i in range(1, num_labels):  # skip background (label 0)
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        # Use the distance at the centroid as approximate radius
        radius = max(int(dist[cy, cx]), int(expected_radius * 0.6))
        
        area_est = np.pi * radius * radius
        if area_est < min_area:
            continue
        
        xmin = max(0, cx - radius)
        ymin = max(0, cy - radius)
        xmax = min(img_w, cx + radius)
        ymax = min(img_h, cy + radius)
        
        balls.append({
            'center': (cx, cy),
            'radius': radius,
            'bbox': (xmin, ymin, xmax, ymax),
            'contour': contour  # keep original contour for reference
        })
    
    return balls


def _remove_duplicate_balls(balls, expected_radius):
    """
    Removes duplicate detections (balls whose centers are too close).
    Keeps the one with the radius closest to expected.
    """
    if len(balls) <= 1:
        return balls
    
    min_dist = expected_radius * 1.2  # minimum distance between ball centers
    
    # Sort by how close radius is to expected (best first)
    balls_sorted = sorted(balls, key=lambda b: abs(b['radius'] - expected_radius))
    
    kept = []
    for ball in balls_sorted:
        cx, cy = ball['center']
        is_duplicate = False
        for existing in kept:
            ex, ey = existing['center']
            dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if dist < min_dist:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(ball)
    
    return kept


# ============================================================
# 4. BALL CLASSIFICATION (color-based)
# ============================================================

def classify_ball(img, ball):
    """
    Classifies a detected ball by its number (0-15) based on color.
    
    Strategy:
        1. Extract the ball region (circular ROI)
        2. Analyze the dominant color in HSV
        3. Determine if solid or stripe (stripes have significant white area)
        4. Map color to ball number
    
    Args:
        img: BGR image
        ball: dict with 'center', 'radius', 'bbox'
    
    Returns:
        number: ball number (0-15)
    """
    cx, cy = ball['center']
    r = ball['radius']
    h_img, w_img = img.shape[:2]
    
    # Create circular mask for this ball
    ball_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(ball_mask, (cx, cy), max(r - 2, 1), 255, -1)
    
    # Extract HSV values within the ball
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ball_pixels = hsv[ball_mask == 255]
    
    if len(ball_pixels) == 0:
        return -1
    
    # --- Check for white (cue ball) ---
    white_mask = (ball_pixels[:, 1] < 40) & (ball_pixels[:, 2] > 180)
    white_ratio = np.sum(white_mask) / len(ball_pixels)
    if white_ratio > 0.7:
        return 0  # Cue ball
    
    # --- Check for black (8 ball) ---
    black_mask = (ball_pixels[:, 2] < 60) & (ball_pixels[:, 1] < 80)
    black_ratio = np.sum(black_mask) / len(ball_pixels)
    if black_ratio > 0.5:
        return 8
    
    # --- Determine if stripe or solid ---
    # Stripes have a white band, so significant white pixels mixed with color
    is_stripe = 0.15 < white_ratio < 0.65
    
    # --- Determine the dominant color (excluding white pixels) ---
    colored_pixels = ball_pixels[~white_mask]
    if len(colored_pixels) == 0:
        return -1
    
    median_h = np.median(colored_pixels[:, 0])
    median_s = np.median(colored_pixels[:, 1])
    median_v = np.median(colored_pixels[:, 2])
    
    # Match to known ball colors
    best_color = None
    for color_name, color_def in BALL_COLORS.items():
        if color_name in ("white", "black"):
            continue
        h_lo, h_hi = color_def["h_range"]
        s_lo, s_hi = color_def["s_range"]
        v_lo, v_hi = color_def["v_range"]
        
        # Handle red wrap-around in hue (0 and 170-180 are both red)
        if color_name in ("red", "maroon"):
            h_match = (median_h <= h_hi) or (median_h >= 170)
        else:
            h_match = h_lo <= median_h <= h_hi
        
        s_match = s_lo <= median_s <= s_hi
        v_match = v_lo <= median_v <= v_hi
        
        if h_match and s_match and v_match:
            best_color = color_name
            break
    
    if best_color is None:
        # Fallback: find closest color by hue distance
        best_color = _closest_color_by_hue(median_h, median_s, median_v)
    
    if best_color is None:
        return -1
    
    color_def = BALL_COLORS[best_color]
    return color_def["stripe"] if is_stripe else color_def["solid"]


def _closest_color_by_hue(h, s, v):
    """Fallback: find closest ball color by hue distance."""
    best_dist = float('inf')
    best_name = None
    
    for name, cdef in BALL_COLORS.items():
        if name in ("white", "black"):
            continue
        h_lo, h_hi = cdef["h_range"]
        h_center = (h_lo + h_hi) / 2
        
        # Circular distance for hue
        dist = min(abs(h - h_center), 180 - abs(h - h_center))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    
    return best_name


# ============================================================
# 5. DRAW BALLS ON TOP-VIEW
# ============================================================

# Color map for drawing each ball number on the top-view
# BGR format
BALL_DRAW_COLORS = {
    0:  (255, 255, 255),  # cue ball - white
    1:  (0, 215, 255),    # yellow
    2:  (255, 50, 50),    # blue
    3:  (0, 0, 220),      # red
    4:  (120, 0, 120),    # purple
    5:  (0, 100, 255),    # orange
    6:  (0, 160, 0),      # green
    7:  (0, 0, 100),      # maroon
    8:  (30, 30, 30),     # black
    9:  (0, 215, 255),    # yellow stripe
    10: (255, 50, 50),    # blue stripe
    11: (0, 0, 220),      # red stripe
    12: (120, 0, 120),    # purple stripe
    13: (0, 100, 255),    # orange stripe
    14: (0, 160, 0),      # green stripe
    15: (0, 0, 100),      # maroon stripe
}


def draw_balls_on_top_view(top_view, raw_balls, classified_numbers, M, img_w, img_h):
    """
    Projects detected ball positions into the top-view using the homography
    matrix M, then draws colored circles with ball numbers.

    Args:
        top_view: the warped top-view image (BGR)
        raw_balls: list of ball dicts with 'center' and 'radius' in original image coords
        classified_numbers: list of int, ball number for each ball in raw_balls
        M: 3x3 perspective transform matrix from original image to top-view
        img_w: original image width (unused here but kept for clarity)
        img_h: original image height

    Returns:
        annotated: top-view image with circles and number labels drawn
    """
    annotated = top_view.copy()
    tv_h, tv_w = annotated.shape[:2]

    for ball, number in zip(raw_balls, classified_numbers):
        cx, cy = ball['center']

        # Project center point into top-view space using M
        pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        pt_warped = cv2.perspectiveTransform(pt, M)
        wx, wy = int(pt_warped[0][0][0]), int(pt_warped[0][0][1])

        # Skip if projected point is outside the top-view canvas
        if not (0 <= wx < tv_w and 0 <= wy < tv_h):
            continue

        # Estimate radius in top-view space by projecting a nearby point
        pt2 = np.array([[[float(cx + ball['radius']), float(cy)]]], dtype=np.float32)
        pt2_warped = cv2.perspectiveTransform(pt2, M)
        wx2 = pt2_warped[0][0][0]
        draw_radius = max(int(abs(wx2 - wx)), 10)

        # Pick drawing color
        color = BALL_DRAW_COLORS.get(number, (200, 200, 200))

        # Draw filled circle
        cv2.circle(annotated, (wx, wy), draw_radius, color, -1)

        # Draw white border to distinguish from cloth
        cv2.circle(annotated, (wx, wy), draw_radius, (255, 255, 255), 1)

        # Draw stripe band for stripe balls (9-15): white horizontal stripe
        if 9 <= number <= 15:
            stripe_y1 = max(wy - draw_radius // 3, 0)
            stripe_y2 = min(wy + draw_radius // 3, tv_h - 1)
            stripe_x1 = max(wx - draw_radius, 0)
            stripe_x2 = min(wx + draw_radius, tv_w - 1)
            # Build a circular clip mask and draw the stripe only inside the circle
            stripe_mask = np.zeros((tv_h, tv_w), dtype=np.uint8)
            cv2.circle(stripe_mask, (wx, wy), draw_radius - 1, 255, -1)
            stripe_region = annotated[stripe_y1:stripe_y2, stripe_x1:stripe_x2]
            mask_region = stripe_mask[stripe_y1:stripe_y2, stripe_x1:stripe_x2]
            stripe_region[mask_region > 0] = (230, 230, 230)

        # Draw the ball number text
        label = str(number)
        font_scale = max(draw_radius / 18.0, 0.35)
        thickness = 1 if draw_radius < 16 else 2
        text_color = (0, 0, 0) if number != 8 else (255, 255, 255)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = wx - tw // 2
        ty = wy + th // 2

        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    return annotated


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_image(image_path):
    """
    Full pipeline for a single image.
    
    Returns:
        result: dict with image_path, num_balls, balls list
        top_view: the warped top-view image with annotated balls (or None)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None, None
    
    h, w = img.shape[:2]
    
    # Step 1: Detect table corners
    corners = detect_table_corners(img)
    if corners is None:
        print(f"[ERROR] Could not detect table in: {image_path}")
        return None, None
    
    # Step 2: Generate top-view
    top_view, M = generate_top_view(img, corners)
    
    # Step 3: Detect balls
    raw_balls = detect_balls(img, corners)
    
    # Step 4: Classify each ball
    result_balls = []
    classified_numbers = []
    for ball in raw_balls:
        number = classify_ball(img, ball)
        classified_numbers.append(number)
        xmin, ymin, xmax, ymax = ball['bbox']
        
        result_balls.append({
            "number": number,
            "xmin": xmin / w,
            "xmax": xmax / w,
            "ymin": ymin / h,
            "ymax": ymax / h,
        })
    
    # Step 5: Draw annotated balls on top-view
    top_view_annotated = draw_balls_on_top_view(top_view, raw_balls, classified_numbers, M, w, h)

    result = {
        "image_path": image_path,
        "num_balls": len(result_balls),
        "balls": result_balls
    }
    
    return result, top_view_annotated


def main():
    """
    Main entry point. Reads input JSON, processes each image,
    writes output JSON and top-view images.
    """
    if len(sys.argv) < 2:
        print("Usage: python pool_detector.py input.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Read input JSON
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    image_paths = input_data["image_path"]
    
    # Create output directory for top-view images
    output_dir = "top_views"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    results = []
    for img_path in image_paths:
        print(f"[INFO] Processing: {img_path}")
        
        result, top_view = process_image(img_path)
        
        if result is not None:
            results.append(result)
        
        if top_view is not None:
            # Save top-view with the same filename
            filename = os.path.basename(img_path)
            top_view_path = os.path.join(output_dir, filename)
            cv2.imwrite(top_view_path, top_view)
            print(f"  -> Top-view saved: {top_view_path}")
        
        if result is not None:
            print(f"  -> Found {result['num_balls']} balls")
    
    # Write output JSON
    output_path = "output.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[DONE] Results saved to {output_path}")
    print(f"[DONE] Top-views saved to {output_dir}/")


if __name__ == "__main__":
    main()
