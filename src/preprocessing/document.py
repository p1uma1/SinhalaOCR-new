import base64
import io

import cv2
import numpy as np
from PIL import Image


def load_and_gray(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def binarize(gray):
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def remove_noise(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def deskew(binary):
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = binary.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def horizontal_projection(binary):
    return np.sum(binary, axis=1)


def vertical_projection(binary):
    return np.sum(binary, axis=0)


def extract_line_positions(binary, threshold_ratio=0.2):
    projection = horizontal_projection(binary)
    max_val = np.max(projection)
    if max_val == 0:
        return []

    threshold = max_val * threshold_ratio
    lines = []
    in_line = False
    start = 0

    for i, value in enumerate(projection):
        if value > threshold and not in_line:
            in_line = True
            start = i
        elif value <= threshold and in_line:
            end = i
            if end - start > 5:
                lines.append((start, end))
            in_line = False

    if in_line:
        lines.append((start, len(projection) - 1))

    return lines


def crop_lines(binary, lines, padding_top=10, padding_bottom=10):
    line_images = []
    h = binary.shape[0]
    for start, end in lines:
        start = max(0, start - padding_top)
        end = min(h, end + padding_bottom)
        line_images.append(binary[start:end, :])
    return line_images


def _merge_boxes_into_words(boxes, max_gap=12):
    """Merge nearby character bounding boxes into word-level boxes (left-to-right)."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    words = []
    current = list(boxes[0])

    for x, y, w, h in boxes[1:]:
        cx, cy, cw, ch = current
        gap = x - (cx + cw)
        if gap <= max_gap:
            nx = min(cx, x)
            ny = min(cy, y)
            nx2 = max(cx + cw, x + w)
            ny2 = max(cy + ch, y + h)
            current = [nx, ny, nx2 - nx, ny2 - ny]
        else:
            words.append(tuple(current))
            current = [x, y, w, h]

    words.append(tuple(current))
    return words


def _boxes_from_connected_components(line_binary, min_area=15):
    """Find text blobs via connected components on the binary line mask."""
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        line_binary, connectivity=8
    )
    boxes = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or w < 2 or h < 2:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _boxes_from_vertical_projection(line_binary, threshold_ratio=0.15, min_width=3):
    """Fallback word boundaries using vertical projection valleys."""
    projection = vertical_projection(line_binary)
    max_val = int(np.max(projection))
    if max_val == 0:
        return []

    threshold = max_val * threshold_ratio
    h = line_binary.shape[0]
    regions = []
    in_word = False
    start = 0

    for i, value in enumerate(projection):
        if value > threshold and not in_word:
            in_word = True
            start = i
        elif value <= threshold and in_word:
            end = i
            if end - start >= min_width:
                y_coords, x_coords = np.where(line_binary[:, start:end] > 0)
                if len(x_coords) == 0:
                    in_word = False
                    continue
                x0 = start + int(x_coords.min())
                x1 = start + int(x_coords.max()) + 1
                y0 = int(y_coords.min())
                y1 = int(y_coords.max()) + 1
                regions.append((x0, y0, x1 - x0, y1 - y0))
            in_word = False

    if in_word:
        end = len(projection)
        if end - start >= min_width:
            y_coords, x_coords = np.where(line_binary[:, start:end] > 0)
            if len(x_coords) > 0:
                x0 = start + int(x_coords.min())
                x1 = start + int(x_coords.max()) + 1
                y0 = int(y_coords.min())
                y1 = int(y_coords.max()) + 1
                regions.append((x0, y0, x1 - x0, y1 - y0))

    return regions


def extract_word_boxes(line_binary, max_gap=12):
    """
    Segment a line binary mask into word-level bounding boxes.
    Uses connected components + gap merging, with vertical projection fallback.
    """
    boxes = _boxes_from_connected_components(line_binary)
    words = _merge_boxes_into_words(boxes, max_gap=max_gap)

    if not words:
        words = _boxes_from_vertical_projection(line_binary)

    if not words:
        h, w = line_binary.shape
        if np.any(line_binary > 0):
            y_coords, x_coords = np.where(line_binary > 0)
            words = [
                (
                    int(x_coords.min()),
                    int(y_coords.min()),
                    int(x_coords.max()) - int(x_coords.min()) + 1,
                    int(y_coords.max()) - int(y_coords.min()) + 1,
                )
            ]

    return words


def crop_masked_text_box(line_binary, bbox, padding=3):
    """
    Crop a text box and apply the binary mask: ink only inside the bbox, white elsewhere.
    Returns dark-on-light image suitable for TrOCR (black text, white background).
    """
    x, y, w, h = bbox
    line_h, line_w = line_binary.shape
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(line_w, x + w + padding)
    y1 = min(line_h, y + h + padding)

    region = line_binary[y0:y1, x0:x1]
    # white background, black ink (TrOCR-friendly polarity)
    masked = np.full(region.shape, 255, dtype=np.uint8)
    masked[region > 0] = 0
    return masked


def normalize_height(img, target_height=64):
    h, w = img.shape
    if h == 0:
        return img
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_height))


def draw_boxes_on_binary(binary, line_spans, word_boxes_by_line):
    """Overlay line (green) and word (red) boxes on deskewed binary for debug."""
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (y0, y1) in line_spans:
        cv2.rectangle(vis, (0, y0), (vis.shape[1] - 1, y1), (0, 255, 0), 1)
    for line_idx, word_boxes in enumerate(word_boxes_by_line):
        if line_idx >= len(line_spans):
            break
        line_y0, _line_y1 = line_spans[line_idx]
        for x, y, w, h in word_boxes:
            cv2.rectangle(
                vis,
                (x, line_y0 + y),
                (x + w, line_y0 + y + h),
                (0, 0, 255),
                1,
            )
    return vis


def _array_to_png_base64(image_array):
    if len(image_array.shape) == 2:
        pil_image = Image.fromarray(image_array)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _read_image_bytes(image_bytes):
    array = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image. Upload a valid PNG or JPEG file.")
    return image_bgr


def extract_lines_from_bytes(image_bytes, return_debug=False, segment_words=True):
    """
    Extract line and word crops from a document image.

    Word boxes are derived from the binary mask; each crop applies that mask so only
  ink inside the box is kept (white background).
    """
    image_bgr = _read_image_bytes(image_bytes)
    gray = load_and_gray(image_bgr)
    blurred = denoise(gray)
    binary = binarize(blurred)
    cleaned = remove_noise(binary)
    deskewed = deskew(cleaned)
    line_positions = extract_line_positions(deskewed)
    line_images = crop_lines(deskewed, line_positions)

    lines = []
    word_boxes_by_line = []
    total_words = 0

    for idx, line_img in enumerate(line_images):
        line_normalized = normalize_height(line_img, 64)
        line_rgb = cv2.cvtColor(line_normalized, cv2.COLOR_GRAY2RGB)

        word_entries = []
        if segment_words:
            word_boxes = extract_word_boxes(line_img)
            word_boxes_by_line.append(word_boxes)

            for word_idx, bbox in enumerate(word_boxes):
                masked = crop_masked_text_box(line_img, bbox)
                masked_norm = normalize_height(masked, 64)
                masked_rgb = cv2.cvtColor(masked_norm, cv2.COLOR_GRAY2RGB)
                word_entries.append(
                    {
                        "index": word_idx,
                        "bbox": list(bbox),
                        "image_base64": _array_to_png_base64(masked_rgb),
                    }
                )
        else:
            word_boxes_by_line.append([])

        total_words += len(word_entries)

        lines.append(
            {
                "index": idx,
                "line_span": list(line_positions[idx]) if idx < len(line_positions) else None,
                "image_base64": _array_to_png_base64(line_rgb),
                "words": word_entries,
                "word_count": len(word_entries),
            }
        )

    result = {
        "line_count": len(lines),
        "word_count": total_words,
        "lines": lines,
        "segmentation": "line+word" if segment_words else "line",
    }

    if return_debug:
        boxed = draw_boxes_on_binary(deskewed, line_positions, word_boxes_by_line)
        result["debug"] = {
            "original": _array_to_png_base64(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)),
            "deskewed": _array_to_png_base64(deskewed),
            "boxes": _array_to_png_base64(boxed),
        }

    return result
