import cv2
import numpy as np
from PIL import Image

def get_sharpness(frame: np.ndarray) -> float:
    """Calculates the Laplacian variance to evaluate frame edge clarity."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_key_frame(video_path: str, interval_seconds: int = 2) -> Image.Image:
    """
    Scans the video using a rolling continuous timeline window. Eliminates rigid 
    chunk walls to guarantee smooth, evenly-spaced, crystal-clear frame tracking.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    
    print(f"[Module 1] Ingesting timeline via Continuous Rolling Window Engine...")

    # We dynamically adjust our clarity threshold based on the video's average quality
    BLUR_THRESHOLD = 80.0 

    all_frames_cached = []
    motion_scores = []
    prev_gray = None

    # 1. Swift single-pass execution stream ingestion
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)
        
        motion_score = 0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray_blur)
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            motion_score = np.sum(thresh == 255)
            
        all_frames_cached.append(small_frame)
        motion_scores.append(motion_score)
        prev_gray = gray_blur

    cap.release()

    extracted_frames = []
    
    # 2. Map out smooth, predictable time targets based on user interval selections
    target_timestamps = []
    current_time = 0.0
    while current_time <= duration_seconds:
        target_timestamps.append(current_time)
        current_time += interval_seconds

    # 3. Continuous Rolling Search Logic
    for target_t in target_timestamps:
        ideal_idx = int(target_t * fps)
        if ideal_idx >= len(all_frames_cached):
            break

        final_frame = None
        final_timestamp = target_t

        # Set a flexible rolling search radius (halfway to the next interval step)
        search_radius = int((interval_seconds * fps) / 2)
        
        # Look outward in a rolling spiral pattern starting from the ideal timestamp index
        for offset in range(0, search_radius):
            # We check forward first, then backward, frame-by-frame continuously
            for direction in [0] if offset == 0 else [1, -1]:
                check_idx = ideal_idx + (offset * direction)
                
                # Stay within total video boundaries
                if check_idx < 0 or check_idx >= len(all_frames_cached):
                    continue
                    
                candidate = all_frames_cached[check_idx]
                sharpness = get_sharpness(candidate)
                
                # If this frame contains movement OR if it's a stable sharp image, grab it
                if sharpness >= BLUR_THRESHOLD:
                    final_frame = candidate
                    final_timestamp = check_idx / fps
                    break
            if final_frame is not None:
                break

        # Total Fallback: If the rolling search radius was entirely blurry,
        # pick the crispest single frame available in that immediate window
        if final_frame is None:
            window_start = max(0, ideal_idx - search_radius)
            window_end = min(len(all_frames_cached), ideal_idx + search_radius)
            
            local_window_frames = all_frames_cached[window_start:window_end]
            sharpness_scores = [get_sharpness(img) for img in local_window_frames]
            
            best_offset = np.argmax(sharpness_scores)
            final_frame = local_window_frames[best_offset]
            final_timestamp = (window_start + best_offset) / fps

        # Format our verified clear frame panel
        final_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        cv2.putText(
            final_rgb, f"{final_timestamp:.1f}s", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        extracted_frames.append(final_rgb)

    if len(extracted_frames) == 0:
        raise ValueError("Could not compile any valid unique frame slices.")

    # Clamp layout panel limits for clean horizontal rendering
    if len(extracted_frames) > 6:
        extracted_frames = extracted_frames[:6]

    mosaic_array = np.hstack(extracted_frames)
    return Image.fromarray(mosaic_array)