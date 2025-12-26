import cv2
import numpy as np

def needs_preprocessing(img, threshold=0.02):
    """
    Detect if image has noisy background or is already clean.
    
    Returns:
        True if image needs preprocessing (has background noise)
        False if image is already clean
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check corners and edges for noise
    # Clean fundus images typically have black/near-black corners
    corner_size = min(h, w) // 10
    
    corners = [
        gray[0:corner_size, 0:corner_size],                    # Top-left
        gray[0:corner_size, w-corner_size:w],                  # Top-right
        gray[h-corner_size:h, 0:corner_size],                  # Bottom-left
        gray[h-corner_size:h, w-corner_size:w],                # Bottom-right
    ]
    
    # Calculate mean intensity of corners
    corner_means = [np.mean(c) for c in corners]
    avg_corner_intensity = np.mean(corner_means)
    
    # Check edge strips
    edge_size = min(h, w) // 15
    edges = [
        gray[0:edge_size, :],           # Top edge
        gray[h-edge_size:h, :],         # Bottom edge
        gray[:, 0:edge_size],           # Left edge
        gray[:, w-edge_size:w],         # Right edge
    ]
    
    edge_means = [np.mean(e) for e in edges]
    avg_edge_intensity = np.mean(edge_means)
    
    # Also check variance in corners (noise has higher variance)
    corner_vars = [np.var(c) for c in corners]
    avg_corner_var = np.mean(corner_vars)
    
    # Decision logic:
    # - Clean images: corners are dark (< 15-20) with low variance
    # - Noisy images: corners have higher intensity or high variance
    
    intensity_threshold = 20  # Corners should be darker than this
    variance_threshold = 500  # Low variance expected for clean black background
    
    has_bright_corners = avg_corner_intensity > intensity_threshold
    has_noisy_corners = avg_corner_var > variance_threshold
    has_bright_edges = avg_edge_intensity > intensity_threshold * 1.5
    
    needs_cleaning = has_bright_corners or has_noisy_corners or has_bright_edges
    
    return needs_cleaning, {
        'corner_intensity': avg_corner_intensity,
        'corner_variance': avg_corner_var,
        'edge_intensity': avg_edge_intensity,
        'decision': 'needs_preprocessing' if needs_cleaning else 'already_clean'
    }


def preprocess_fundus(image_path, output_path=None, padding=5, force=False):
    """
    Preprocess fundus image: crop to retinal region and remove background noise.
    Only processes images that need it (unless force=True).
    
    Args:
        image_path: Path to input fundus image
        output_path: Path to save processed image (optional)
        padding: Pixels to add around detected region (default: 5)
        force: If True, always apply preprocessing regardless of image quality
    
    Returns:
        Processed image with black background
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Check if preprocessing is needed
    if not force:
        needs_proc, stats = needs_preprocessing(img)
        if not needs_proc:
            print(f"Image already clean, skipping preprocessing. Stats: {stats}")
            if output_path:
                cv2.imwrite(output_path, img)
            return img
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(thresh) > 200:
        _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contours found, returning original image")
        return img
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        cv2.ellipse(mask, ellipse, 255, -1)
    else:
        hull = cv2.convexHull(largest_contour)
        cv2.drawContours(mask, [hull], -1, 255, -1)
    
    combined_mask = cv2.bitwise_and(thresh, mask)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=2)
    
    result = cv2.bitwise_and(original, original, mask=combined_mask)
    
    mask_points = cv2.findNonZero(combined_mask)
    if mask_points is None:
        print("Warning: Empty mask, returning original")
        return img
    
    x, y, w, h = cv2.boundingRect(mask_points)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(result.shape[1] - x, w + 2 * padding)
    h = min(result.shape[0] - y, h + 2 * padding)
    
    cropped = result[y:y+h, x:x+w]
    
    if output_path:
        cv2.imwrite(output_path, cropped)
        print(f"Saved preprocessed image to: {output_path}")
    
    return cropped


def preprocess_fundus_robust(image_path, output_path=None, padding=5, force=False):
    """
    More robust version using multiple detection strategies.
    Only processes images that need it (unless force=True).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Check if preprocessing is needed
    if not force:
        needs_proc, stats = needs_preprocessing(img)
        if not needs_proc:
            print(f"Image already clean, skipping. Stats: {stats}")
            if output_path:
                cv2.imwrite(output_path, img)
            return img
        print(f"Preprocessing needed. Stats: {stats}")
    
    original = img.copy()
    h, w = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    combined = cv2.addWeighted(gray, 0.5, v_channel, 0.5, 0)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    
    masks = []
    for thresh_val in [10, 15, 20, 25]:
        _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        masks.append(mask)
    
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(otsu_mask)
    
    expected_area = np.pi * (min(h, w) / 2) ** 2 * 0.7
    
    best_mask = None
    best_diff = float('inf')
    
    for mask in masks:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        
        area = cv2.countNonZero(mask_clean)
        diff = abs(area - expected_area)
        
        if diff < best_diff and area > expected_area * 0.3:
            best_diff = diff
            best_mask = mask_clean
    
    if best_mask is None:
        best_mask = masks[2]
    
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    final_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    result = cv2.bitwise_and(original, original, mask=final_mask)
    
    mask_points = cv2.findNonZero(final_mask)
    bx, by, bw, bh = cv2.boundingRect(mask_points)
    
    bx = max(0, bx - padding)
    by = max(0, by - padding)
    bw = min(w - bx, bw + 2 * padding)
    bh = min(h - by, bh + 2 * padding)
    
    cropped = result[by:by+bh, bx:bx+bw]
    
    if output_path:
        cv2.imwrite(output_path, cropped)
        print(f"Saved: {output_path}")
    
    return cropped


def preprocess_dataset(input_dir, output_dir, method='robust', force=False):
    """
    Process all images in a directory.
    Automatically skips clean images unless force=True.
    
    Args:
        input_dir: Directory containing fundus images
        output_dir: Directory to save processed images
        method: 'basic' or 'robust'
        force: If True, process all images regardless of quality
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    func = preprocess_fundus_robust if method == 'robust' else preprocess_fundus
    
    extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    
    stats = {'processed': 0, 'skipped': 0, 'errors': 0}
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                img = cv2.imread(input_path)
                needs_proc, _ = needs_preprocessing(img)
                
                if needs_proc or force:
                    func(input_path, output_path, force=force)
                    stats['processed'] += 1
                    print(f"Processed: {filename}")
                else:
                    cv2.imwrite(output_path, img)
                    stats['skipped'] += 1
                    print(f"Skipped (clean): {filename}")
                    
            except Exception as e:
                stats['errors'] += 1
                print(f"Error processing {filename}: {e}")
    
    print(f"\nSummary: {stats['processed']} processed, {stats['skipped']} skipped, {stats['errors']} errors")
    return stats


# Example usage
if __name__ == "__main__":
    # Single image - auto-detects if preprocessing needed
    # result = preprocess_fundus_robust(r"C:\Users\aryan\Projects\Major\data\images\Normal\AMDnet23_Normal_20_left.jpeg",
    #  "./output.jpg")
    
    # Force preprocessing even on clean images
    # result = preprocess_fundus_robust("fundus_image.jpg", "output.jpg", force=True)
    
    # Batch processing - auto-skips clean images
    preprocess_dataset(r"C:\Users\aryan\Projects\Major\data\images\AMD",
                        r"C:\Users\aryan\Projects\Major\preproc\AMD",
                         method='robust')