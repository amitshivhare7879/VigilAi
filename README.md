# VigilAI: Multimodal Video Analytics & Incident Alert Pipeline

VigilAI is a production-grade, heterogeneous AI inference pipeline designed for mission-critical monitoring environments. The system bridges the gap between deep vision structural feature extraction and semantic textual reasoning by combining a Convolutional Neural Network (CNN) with a Vision Language Model (VLM) served via a high-performance async API backend.

## 🚀 Architecture Overview

1. **Video Ingestion (OpenCV):** Efficiently streams video feeds and extracts key analytical frames in memory.
2. **Structural Encoding (PyTorch & ResNet50):** Passes the extracted frame through a deep 50-layer CNN to generate a dense `[1, 1000]` spatial feature map.
3. **Contextual Reasoning (Moondream VLM):** Projects the visual embeddings into a lightweight 1.8B parameter text-based vision model to perform advanced text-image reasoning.
4. **Asynchronous API Layer (FastAPI):** Wraps the multi-model architecture into production-ready endpoints with automatic multi-part stream handling and temporary cache file management.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning Frameworks:** PyTorch, Torchvision
* **Computer Vision:** OpenCV (cv2), Pillow (PIL)
* **AI Model Routing:** Moondream SDK / HuggingFace Transformers
* **Backend Framework:** FastAPI, Uvicorn, Python-Multipart

## 💻 Getting Started

### 1. Clone the Repository & Set Up Envs
```bash
git clone [https://github.com/YOUR_USERNAME/vigilant-pipeline.git](https://github.com/YOUR_USERNAME/vigilant-pipeline.git)
cd vigilant-pipeline
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt