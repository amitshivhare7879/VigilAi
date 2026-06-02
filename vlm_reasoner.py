import moondream as md
from PIL import Image
import os

# Import Module 1 to fetch our video frames dynamically
from video_handler import extract_key_frame

class VLMReasoner:
    def __init__(self):
        print("[Module 3] Initializing Moondream VLM Engine via Cloud API...")
        
        # Paste your real Moondream API key inside the quotes below:
        self.model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiIxMDQ3YzU3NC00ZjBlLTRlNjYtODdhMC1kMTJjZDlmMmE0N2UiLCJvcmdfaWQiOiJzbzB5ckZwZllQdTd2djlGNXhWSU95aE5oVmwxZTlxayIsImlhdCI6MTc4MDMwOTk5OCwidmVyIjoxfQ.jH03W4EhxEXQpjjTzK0RzhToJnfHLBk5O5sDHQDn3II")
        
        print("[Module 3] VLM successfully authenticated and ready.")

    def analyze_image_context(self, pil_image: Image.Image, prompt: str) -> str:
        """
        Encodes the image properties and queries the VLM using the authenticated API key.
        """
        print("[Module 3] Encoding visual frame tensors for VLM projection...")
        encoded_image = self.model.encode_image(pil_image)
        
        print(f"[Module 3] Executing textual reasoning with prompt: '{prompt}'")
        response = self.model.query(encoded_image, prompt)
        
        return response["answer"].strip()

# --- Full Core Inference Pipeline Verification ---
if __name__ == "__main__":
    target_video = "test.mp4"
    
    if os.path.exists(target_video):
        print("\n--- Testing Full End-to-End Core Pipeline ---")
        try:
            # 1. Extract image frame using Module 1
            img_frame = extract_key_frame(target_video)
            
            # 2. Initialize the VLM engine
            reasoner = VLMReasoner()
            
            # 3. Query the pipeline locally
            test_prompt = "Describe what is happening in this scene clearly."
            analysis_result = reasoner.analyze_image_context(img_frame, test_prompt)
            
            print("\n================= ANOMALY / INCIDENT REPORT =================")
            print(analysis_result)
            print("=============================================================")
            
        except Exception as e:
            print(f"[Module 3 Pipeline Error]: {e}")
    else:
        print(f"[Error] Missing '{target_video}' in workspace root directory.")