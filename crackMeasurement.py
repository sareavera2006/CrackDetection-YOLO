import cv2
import numpy as np
from skimage.morphology import skeletonize

class CrackAnalysis:

    def __init__(self, pixels_per_mm=None):

        self.pixels_per_mm = pixels_per_mm

    def analyzeCrackPixels(self, image_path, threshold_method='otsu', pixels_per_mm=None):
        """
        Analyzes a crack image and measures its dimensions in pixels.
        If `pixels_per_mm` is provided, also calculates measurements in millimeters.

        :param image_path: Path to the target image
        :param threshold_method: 'otsu' or 'adaptive' binary thresholding.
        :param pixels_per_mm: Scaling factor (px/mm). Defaults to None (no effect).

        :return results:
        """
        # Load image and convert to grayscale
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Smooth surface texture while preserving sharp crack edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Segment dark crack pixels from the lighter background surface
        if threshold_method == 'otsu':

            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Adaptive thresholding handles non-uniform ambient lighting across the surface
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 3
            )

        # 4. Morphological noise reduction (removes isolated noise dots)
        kernel = np.ones((3, 3), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 5. Measure Total Area (Count of all white pixels in the mask)
        crack_area_pixels = int(np.count_nonzero(binary_clean))

        # 6. Skeletonization (Reduces crack to 1-pixel wide centerline for path length measurement)
        binary_bool = binary_clean > 0
        skeleton = skeletonize(binary_bool)
        crack_length_pixels = float(np.count_nonzero(skeleton))

        # Distance Transform (Finds shortest distance from inside the crack to its boundary)
        dist_transform = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)

        # Maximum width = 2 * peak distance value inside the crack
        max_radius = float(np.max(dist_transform)) if np.max(dist_transform) > 0 else 0.0
        max_width_pixels = max_radius * 2.0

        # Average width measured along the skeleton centerline
        skeleton_distances = dist_transform[skeleton]
        avg_width_pixels = float(np.mean(skeleton_distances) * 2.0) if len(skeleton_distances) > 0 else 0.0

        # Create visual overlay output
        visualization = img.copy()
        visualization[binary_clean > 0] = [255, 100, 0]  # Highlight crack mask in blue
        visualization[skeleton] = [0, 0, 255]  # Highlight skeleton path in red

        # Base pixel results
        results = {
            "crack_area_pixels": crack_area_pixels,
            "crack_length_pixels": round(crack_length_pixels, 2),
            "max_width_pixels": round(max_width_pixels, 2),
            "avg_width_pixels": round(avg_width_pixels, 2),
            "visualization": visualization
        }

        # Physical unit conversion (only executes if pixels_per_mm is set)
        if pixels_per_mm is not None and pixels_per_mm > 0:
            results["crack_area_mm2"] = round(crack_area_pixels / (pixels_per_mm ** 2), 2)
            results["crack_length_mm"] = round(crack_length_pixels / pixels_per_mm, 2)
            results["max_width_mm"] = round(max_width_pixels / pixels_per_mm, 2)
            results["avg_width_mm"] = round(avg_width_pixels / pixels_per_mm, 2)

        return results

