import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from src.config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, TARGET_SIZE

def segment_letters(image_array):
    points = np.column_stack(np.where(image_array <= 100))
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
    labels = clustering.labels_
    letter_images = []
    centers = []
    h, w = image_array.shape
    for label in sorted(set(labels)):
        if label == -1:
            continue
        pts = points[labels == label]
        ys, xs = pts[:, 0], pts[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        width = max_x - min_x
        height = max_y - min_y
        dim = int(max(width, height))
        pad = dim // 3
        dim = dim + 2 * pad
        x1 = int(min_x - pad)
        y1 = int(min_y - pad)
        x2 = x1 + dim
        y2 = y1 + dim
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w, x2), min(h, y2)
        crop = image_array[y1_c:y2_c, x1_c:x2_c]
        inv = cv2.bitwise_not(crop)
        resized = cv2.resize(inv, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        norm = resized / 255.0
        norm = norm.reshape(TARGET_SIZE[0], TARGET_SIZE[1], 1).astype('float32')
        letter_images.append(norm)
        centers.append(min_x + width / 2)
    return letter_images, centers