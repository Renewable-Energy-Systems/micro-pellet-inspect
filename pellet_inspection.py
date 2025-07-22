import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def select_camera():
    """
    Scans for available cameras, lists them, and prompts the user to select one.

    Returns:
        The integer index of the selected camera, or None if no camera is selected.
    """
    available_cameras = []
    # Check for cameras by iterating through indices (e.g., 0 to 9)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        logging.error("No cameras found.")
        return None
        
    print("Available Cameras:")
    for idx, cam_index in enumerate(available_cameras):
        print(f"{idx}: Camera Index {cam_index}")

    while True:
        try:
            choice = int(input("Please select a camera by number: "))
            if 0 <= choice < len(available_cameras):
                return available_cameras[choice]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\nSelection cancelled.")
            return None


def inspect_pellet(img, image_debug=False):
    """
    Inspects a pellet from an image for roundness, cracks, and surface finish.
    
    Args:
        img: The input image frame from the webcam.
        image_debug (bool): If True, displays intermediate processing images.

    Returns:
        A new image with inspection results drawn on it.
    """
    if img is None:
        logging.error("Image not loaded.")
        return None

    # --- Pre-processing ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Adaptive thresholding is robust for varying lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 4)

    # --- Find Pellet Contour ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # logging.info("No pellet detected.") # This can be noisy, so it's commented out
        return img  # Return the original image if no contour is found

    # Assume the largest contour is the pellet
    pellet_contour = max(contours, key=cv2.contourArea)
    pellet_area = cv2.contourArea(pellet_contour)

    # Filter out very small contours to avoid noise
    if pellet_area < 500:
        return img

    # Create a mask for the pellet's surface area
    pellet_mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(pellet_mask, [pellet_contour], -1, 255, -1)

    # --- 1. Roundness Inspection ---
    perimeter = cv2.arcLength(pellet_contour, True)
    # Circularity formula: 4*pi*area / perimeter^2. A perfect circle has a value of 1.0
    circularity = (4 * np.pi * pellet_area) / (perimeter ** 2) if perimeter > 0 else 0
    roundness_text = f"Roundness: {circularity:.3f}"
    cv2.putText(img, roundness_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.drawContours(img, [pellet_contour], -1, (0, 255, 0), 2)

    # --- 2. Crack Detection on Outer Diameter (OD) ---
    # Analyze the contour's deviation from its convex hull
    hull = cv2.convexHull(pellet_contour)
    hull_area = cv2.contourArea(hull)
    solidity = pellet_area / float(hull_area) if hull_area > 0 else 0
    # Significant deviations (low solidity) can indicate OD cracks or chips
    if solidity < 0.95:
        cv2.putText(img, "OD Defect Detected", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 3. Crack Detection on Pellet Surface ---
    # Use a black-hat morphological transform to find dark spots (cracks) on a bright background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Apply the mask to only look for cracks inside the pellet
    blackhat_masked = cv2.bitwise_and(blackhat, blackhat, mask=pellet_mask)
    # Threshold the result to find significant cracks
    _, crack_thresh = cv2.threshold(blackhat_masked, 40, 255, cv2.THRESH_BINARY)
    
    # Find contours of the detected cracks
    crack_contours, _ = cv2.findContours(crack_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if crack_contours:
        cv2.putText(img, "Surface Crack Detected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Draw the detected cracks in red
        cv2.drawContours(img, crack_contours, -1, (0, 0, 255), 1)

    # --- 4. Surface Finish Inspection ---
    # Use the standard deviation of pixel intensity inside the pellet as a proxy for roughness
    _, std_dev_val = cv2.meanStdDev(gray, mask=pellet_mask)
    surface_text = f"Surface Roughness (StdDev): {std_dev_val[0][0]:.2f}"
    cv2.putText(img, surface_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if image_debug:
        # For debugging, you can show intermediate processing steps
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Crack Mask", crack_thresh)

    return img

if __name__ == "__main__":
    camera_index = select_camera()

    if camera_index is None:
        exit()

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logging.error(f"Cannot open camera at index {camera_index}")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame is not read correctly, skip the frame and continue
        if not ret:
            logging.warning("Can't receive frame. Skipping...")
            continue

        # Run the inspection on the current frame
        processed_frame = inspect_pellet(frame, image_debug=True)
        
        # Check if the processed frame is valid before displaying
        if processed_frame is not None:
            cv2.imshow('Pellet Inspection', processed_frame)
        else:
            # If processing fails, show the original frame
            cv2.imshow('Pellet Inspection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()