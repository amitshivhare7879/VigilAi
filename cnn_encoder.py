import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Import Module 1 directly from your video_handler.py file
from video_handler import extract_key_frame

class CNNEncoder:
    def __init__(self):
        print("[Module 2] Initializing ResNet50 CNN Backbone...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Module 2] Running CNN on device: {self.device}")
        
        # Load pre-trained ResNet50 weights
        self.weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=self.weights).to(self.device)
        self.model.eval()
        
        # Transformation pipeline
        self.transform_pipeline = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def extract_features(self, pil_image: Image.Image):
        """Transforms a PIL image into mathematical structural feature maps."""
        input_tensor = self.transform_pipeline(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor)
            
        return features

# --- Connected Pipeline Verification Test ---
if __name__ == "__main__":
    import os
    
    # We will use the 'test.mp4' file visible in your VS Code workspace
    target_video = "test.mp4"
    
    if os.path.exists(target_video):
        print(f"\n--- Testing Interconnected Pipeline using '{target_video}' ---")
        try:
            # 1. Dynamically fetch the image frame using Module 1 in memory
            img = extract_key_frame(target_video)
            
            # 2. Instantiate our deep learning backbone
            encoder = CNNEncoder()
            
            # 3. Extract features directly
            raw_vector = encoder.extract_features(img)
            
            print("\n[Module 2] Success! In-memory pipeline executed perfectly.")
            print(f"[Module 2] Raw feature vector shape: {list(raw_vector.shape)}")
            print(f"[Module 2] Sample values from the image matrix:\n{raw_vector[0][:5].tolist()} ...")
            
        except Exception as e:
            print(f"[Pipeline Error]: {e}")
    else:
        print(f"[Error] Could not locate '{target_video}' in your folder. Double check your file path!")