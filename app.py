from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os

# Import the core logic modules we just successfully built and tested
from video_handler import extract_key_frame
from cnn_encoder import CNNEncoder
from vlm_reasoner import VLMReasoner

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="VigilAI / OmniSight Multimodal Backend",
    description="Production-grade AI pipeline extracting CNN visual tensors and auto-generating text incident logs via a local/API VLM."
)

print("[Module 4] Initializing and caching deep learning models onto server startup...")
# 2. Instantiate our deep learning modules globally so they stay cached in memory
cnn_backbone = CNNEncoder()
vlm_engine = VLMReasoner()
print("[Module 4] Web Server is fully optimized and ready to accept traffic.")


@app.post("/analyze-video/", tags=["Inference Pipelines"])
async def analyze_video(
    file: UploadFile = File(...), 
    prompt: str = "Is there any danger, threat, or operational issue in this video frame? Explain clearly."
):
    """
    Accepts an uploaded video file via an HTTP POST request, passes it through the 
    in-memory multi-model pipeline, and returns structural features + natural language logs.
    """
    # 1. Basic guardrail: Verify the user uploaded a video format
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a valid video file (.mp4, .avi, .mov).")

    # 2. Save the uploaded byte stream to a temporary storage file safely
    temp_file_path = f"server_temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Fire up Module 1: Extract the middle frame in-memory
        img_frame = extract_key_frame(temp_file_path)
        
        # 4. Fire up Module 2: Run the CNN feature extraction matrix
        structural_features = cnn_backbone.extract_features(img_frame)
        
        # 5. Fire up Module 3: Let the VLM reason textually over the frame tokens
        incident_report = vlm_engine.analyze_image_context(img_frame, prompt)
        
        # 6. Compile everything into a structured JSON payload response
        return {
            "status": "Success",
            "filename": file.filename,
            "cnn_tensor_shape": list(structural_features.shape),
            "ai_analysis_report": incident_report
        }

    except Exception as e:
        print(f"[Server Application Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
        
    finally:
        # 7. Crucial Production Step: Clean up the temporary system file streams 
        # so your server hard drive doesn't run out of storage space.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)