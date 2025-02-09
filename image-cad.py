import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import ezdxf

def consolidate_with_shapely(outer_contours, inner_contours, img_shape):
    """
    Consolidate outer and inner contours separately and then subtract inner from outer.
    Returns:
      - consolidated_img: a binary image with the consolidated boundaries drawn.
      - boundaries: a list of numpy arrays (each of shape (N,2)) representing the boundary coordinates.
    """
    outer_polys = []
    for cnt in outer_contours:
        pts = cnt[:, 0, :]
        if len(pts) >= 3:
            poly = Polygon(pts).buffer(0)  # buffer(0) can fix slight self-intersections
            if poly.is_valid:
                outer_polys.append(poly)
    
    inner_polys = []
    for cnt in inner_contours:
        pts = cnt[:, 0, :]
        if len(pts) >= 3:
            poly = Polygon(pts).buffer(0)
            if poly.is_valid:
                inner_polys.append(poly)
    
    if outer_polys:
        outer_union = unary_union(outer_polys)
    else:
        outer_union = None

    if inner_polys:
        inner_union = unary_union(inner_polys)
    else:
        inner_union = None

    # Subtract inner features from the outer union if possible.
    if outer_union is None:
        final_geom = None
    elif inner_union is not None:
        final_geom = outer_union.difference(inner_union)
    else:
        final_geom = outer_union

    # Create a blank image to draw the boundaries.
    consolidated_img = np.zeros(img_shape, dtype=np.uint8)
    
    # Extract boundaries from the resulting geometry.
    boundaries = []
    def extract_boundaries(geom):
        if geom.geom_type == 'Polygon':
            boundaries.append(np.array(geom.exterior.coords, dtype=np.int32))
            for interior in geom.interiors:
                boundaries.append(np.array(interior.coords, dtype=np.int32))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom:
                boundaries.append(np.array(poly.exterior.coords, dtype=np.int32))
                for interior in poly.interiors:
                    boundaries.append(np.array(interior.coords, dtype=np.int32))
    
    if final_geom is not None and not final_geom.is_empty:
        extract_boundaries(final_geom)
    
    # Draw all boundaries on the consolidated image.
    for boundary in boundaries:
        cv2.polylines(consolidated_img, [boundary], isClosed=True, color=255, thickness=1)
    
    return consolidated_img, boundaries

def export_boundaries_to_dxf(boundaries, dxf_path):
    """
    Export a list of boundaries (each a numpy array of points) to a DXF file.
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    
    for boundary in boundaries:
        # Convert each point (x, y) to a tuple.
        points = [tuple(pt) for pt in boundary]
        # Add a lightweight polyline; set closed=True so that the last point connects to the first.
        msp.add_lwpolyline(points, dxfattribs={'closed': True})
    
    doc.saveas(dxf_path)
    print(f"DXF file saved as: {dxf_path}")

def precision_edge_detection(input_path, output_path, dxf_path):
    # Load image and verify
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Image not loaded")
        return

    # Create debug window
    cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
    
    # 1. Whiteboard Calibration (Show this first)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    background_mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow("Debug", background_mask)
    cv2.waitKey(0)
    
    # 2. Component Isolation (Show mask refinement)
    component_mask = cv2.bitwise_not(background_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("Debug", component_mask)
    cv2.waitKey(0)
    
    # 3. Enhanced Processing (Show threshold stages)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (7,7), 0)
    
    # Define constants
    MIN_FEATURE_AREA = 50  # Minimum area for inner features
    MEDIAN_BLUR_SIZE = 1   # Kernel size for median blur; must be odd
    if MEDIAN_BLUR_SIZE % 2 == 0:
        MEDIAN_BLUR_SIZE += 1
    
    # Dual thresholding
    _, th_global = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 4)
    combined = cv2.bitwise_or(th_global, th_adaptive)
    cv2.imshow("Debug", combined)
    cv2.waitKey(0)
    
    # 4. Contour Processing with Hierarchy
    all_contours, hierarchy = cv2.findContours(combined.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    
    # Separate outer and inner contours
    outer_contours = []
    inner_features = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, (cnt, hier) in enumerate(zip(all_contours, hierarchy)):
            if hier[3] == -1:
                outer_contours.append(cnt)
            else:
                if cv2.contourArea(cnt) > MIN_FEATURE_AREA:
                    inner_features.append(cnt)
    
    # Create an output image with the raw drawn contours (for debugging)
    raw_output = np.zeros_like(gray)
    cv2.drawContours(raw_output, outer_contours, -1, 255, 2)
    cv2.drawContours(raw_output, inner_features, -1, 255, 1)
    cv2.imshow("Debug", raw_output)
    cv2.waitKey(0)
    
    # 5. Optional Smoothing
    if MEDIAN_BLUR_SIZE > 1:
        smoothed = cv2.medianBlur(raw_output, MEDIAN_BLUR_SIZE)
    else:
        smoothed = raw_output.copy()
    _, smoothed = cv2.threshold(smoothed, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow("Debug", smoothed)
    cv2.waitKey(0)
    
    # 6. Shapely-based Geometric Consolidation (Preserving Inner Features)
    consolidated, boundaries = consolidate_with_shapely(outer_contours, inner_features, gray.shape)
    cv2.imshow("Debug", consolidated)
    cv2.waitKey(0)
    
    cv2.imwrite(output_path, consolidated)
    print(f"Consolidated image saved as: {output_path}")
    
    # 7. Export boundaries to DXF for CAD measurement
    export_boundaries_to_dxf(boundaries, dxf_path)
    
    cv2.destroyAllWindows()

# Usage:
# Replace "oip.jpg" with your input image filename.
precision_edge_detection("oip.jpg", "refined_output.png", "refined_output.dxf")
