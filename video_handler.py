import cv2
import numpy as np
from PIL import Image

def extract_key_frame(video_path: str) -> Image.Image:
    """
    Extracts 3 distinct chronological frames (25%, 50%, 75%) from the video
    and stitches them horizontally into a single mosaic strip for complete coverage.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open or find video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Module 1] Processing multi-frame array across {total_frames} frames...")

    # Calculate exact frame indices for 25%, 50%, and 75% marks
    target_indices = [int(total_frames * 0.25), int(total_frames * 0.50), int(total_frames * 0.75)]
    extracted_frames = []

    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            # Resize individual frames down to a uniform 400x300 size to prevent massive image sizes
            resized_frame = cv2.resize(frame, (400, 300))
            # Convert BGR to RGB layout
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            extracted_frames = extracted_frames + [rgb_frame]

    cap.release()

    if len(extracted_frames) == 0:
        raise ValueError("Failed to extract any valid frame states from video asset.")

    # Stitch the 3 crisp frames horizontally (side-by-side) into a single canvas matrix
    mosaic_array = np.hstack(extracted_frames)
    
    # Convert the complete mosaic array into a single PIL Image
    mosaic_image = Image.fromarray(mosaic_array)
    print("[Module 1] Chronological 3-frame mosaic strip successfully generated.")
    return mosaic_image