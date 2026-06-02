import cv2
import numpy as np
from PIL import Image

def extract_key_frame(video_path: str, interval_seconds: int = 2) -> Image.Image:
    """
    Computes a baseline background image, then samples the video chronologically.
    If a sampled frame is a duplicate or completely static compared to the background, 
    the engine dynamically searches backwards or forwards to find the next meaningful movement.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    
    print(f"[Module 1] Processing video. Duration: {duration_seconds:.2f}s")

    # 1. Generate a median background baseline to compare frames against
    # This helps us identify what the "normal empty room" looks like.
    frame_indices = np.linspace(0, total_frames - 1, min(15, total_frames), dtype=int)
    background_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if success:
            background_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
    if len(background_frames) > 0:
        median_background = np.median(background_frames, axis=0).astype(dtype=np.uint8)
        median_background = cv2.GaussianBlur(median_background, (15, 15), 0)
    else:
        median_background = None

    # 2. Map target second timestamps based on user selection
    current_time = 0.0
    target_times = []
    while current_time <= duration_seconds:
        target_times.append(current_time)
        current_time += interval_seconds

    extracted_frames = []
    
    # 3. Dynamic Bi-directional search loop
    for t in target_times:
        base_frame_idx = int(t * fps)
        found_frame = None
        
        # Search window radius: allows looking backward up to half the interval step
        max_search_frames = int((interval_seconds * fps) / 2)
        
        # We check the exact timestamp frame, then search outward (-1, +1, -2, +2...)
        for offset in range(0, max_search_frames):
            # Check backward first, then forward
            for direction in [-1, 1] if offset > 0 else [0]:
                check_idx = base_frame_idx + (offset * direction)
                
                # Boundary check
                if check_idx < 0 or check_idx >= total_frames:
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, check_idx)
                success, frame = cap.read()
                if not success:
                    continue
                    
                # Evaluate if this frame contains a distinct visual change from the ambient background
                if median_background is not None:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_blur = cv2.GaussianBlur(gray_frame, (15, 15), 0)
                    
                    # Absolute delta math against baseline background
                    diff = cv2.absdiff(median_background, gray_blur)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    change_score = np.sum(thresh == 255) / (frame.shape[0] * frame.shape[1]) * 100
                    
                    # If there's clear deviation (something new appeared), lock this frame
                    if change_score > 1.5: 
                        found_frame = frame
                        actual_time = check_idx / fps
                        break
            if found_frame is not None:
                break
                
        # Fallback: If the whole window is static, just take the default timestamp frame
        if found_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(base_frame_idx, total_frames - 1))
            _, found_frame = cap.read()
            actual_time = t

        if found_frame is not None:
            resized = cv2.resize(found_frame, (320, 240))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Label with the precise runtime location
            cv2.putText(
                rgb_frame, f"{actual_time:.1f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
            extracted_frames.append(rgb_frame)

    cap.release()

    if len(extracted_frames) == 0:
        raise ValueError("Could not compile any unique frame slices.")

    # Trim to avoid breaking wide layouts
    if len(extracted_frames) > 6:
        extracted_frames = extracted_frames[:6]

    mosaic_array = np.hstack(extracted_frames)
    return Image.fromarray(mosaic_array)