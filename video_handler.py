import cv2
from PIL import Image

def extract_key_frame(video_path: str) -> Image.Image:
    """
    Opens a video file, navigates to the midpoint frame, 
    and converts it into a PIL Image for AI processing.
    """
    # 1. Connect to the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open or find the video: {video_path}")

    # 2. Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Module 1] Total frames detected: {total_frames}")

    # 3. Target the middle frame (excellent proxy for a short clip's content)
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    # 4. Read the frame
    success, frame = cap.read()
    cap.release() # Always close the video file stream immediately

    if not success:
        raise ValueError("Failed to extract frame from the video track.")

    # 5. Crucial Step: OpenCV reads images in BGR format, 
    # but PyTorch/VLMs expect RGB format. We must swap them!
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 6. Convert the numpy array into a PIL Image object
    pil_image = Image.fromarray(rgb_frame)
    print(f"[Module 1] Successfully extracted middle frame ({middle_frame_index}) as PIL Image.")
    
    return pil_image

# --- Local Verification Test ---
if __name__ == "__main__":
    # To test this module alone, place a small video file named "test.mp4" 
    # in your folder and uncomment the lines below:
    #
    try:
        img = extract_key_frame("test.mp4")
        img.show() # This opens the extracted frame on your desktop screen
    except Exception as e:
        print(f"Error: {e}")
    pass