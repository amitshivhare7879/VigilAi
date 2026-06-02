import streamlit as st
import os
import sys

# Import your cleanly separated modules directly into the frontend state
from video_handler import extract_key_frame
from cnn_encoder import CNNEncoder
from vlm_reasoner import VLMReasoner

# =========================================================
# 1. STREAMLIT FRONTEND DESIGN LAYER
# =========================================================
def run_ui_dashboard():
    st.set_page_config(page_title="VigilAI Core Dashboard", page_icon="🛡️", layout="wide")

    st.title("🛡️ VigilAI: Modular Video Portal")
    st.caption("Architecture: Completely separate custom tracking modules running concurrently inside your Python environment runtime.")
    st.markdown("---")

    # Cache your deep learning backend classes globally so they load only once
    @st.cache_resource
    def load_pipeline_components():
        vision = CNNEncoder()
        reasoner = VLMReasoner()
        return vision, reasoner

    try:
        cnn_backbone, vlm_engine = load_pipeline_components()
    except Exception as e:
        st.error(f"Initialization Fault: Ensure your API key is correctly configured inside vlm_reasoner.py. ({e})")
        st.stop()

    # Layout Configuration
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload Input Stream Assets")
        uploaded_file = st.file_uploader("Drop operational video feed here...", type=["mp4", "avi", "mov", "mkv"])
        custom_prompt = st.text_area("🧠 AI Query Directives", value="Is there any danger, threat, or operational issue in this video frame? Explain clearly.")
        trigger_analysis = st.button("🚀 Execute In-Memory Pipeline", use_container_width=True)

    with col2:
        st.subheader("📊 Pipeline Diagnostics Engine")
        
        if trigger_analysis and uploaded_file is not None:
            # Create a temporary local video copy to pass into OpenCV paths safely
            temp_name = f"ui_temp_cache_{uploaded_file.name}"
            with open(temp_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            status_bar = st.empty()
            progress = st.progress(0)
            
            try:
                # Execution Node 1: Extract frame array via Module 1
                status_bar.info("⚙️ Step 1/3: Slicing video track stream arrays via Module 1...")
                progress.progress(25)
                pil_frame = extract_key_frame(temp_name)
                
                # Render the sampled analytical frame dynamically onto the frontend page
                st.image(pil_frame, caption="Module 1: Sampled Contextual Frame Target", use_container_width=True)
                
                # Execution Node 2: Run CNN embedding arrays via Module 2
                status_bar.info("⚙️ Step 2/3: Generating structural ResNet50 feature maps via Module 2...")
                progress.progress(60)
                structural_features = cnn_backbone.extract_features(pil_frame)
                
                # Execution Node 3: Generate text output via Module 3
                status_bar.info("⚙️ Step 3/3: Evaluating multimodal text token mappings via Module 3...")
                progress.progress(85)
                ai_report = vlm_engine.analyze_image_context(pil_frame, custom_prompt)
                
                # Finalize Pipeline display states
                progress.progress(100)
                status_bar.success("✅ Multi-Module pipeline successfully processed.")
                
                st.markdown("### 📝 Generated Incident Log Report")
                st.success(ai_report)
                
                with st.expander("🛠️ View Isolated Module Metas"):
                    st.json({
                        "Source Target Asset": uploaded_file.name,
                        "Module 2 Tensor Layer Shape Output": list(structural_features.shape)
                    })
                    
            except Exception as err:
                status_bar.error(f"Critical execution block fault: {err}")
                
            finally:
                # Security: Wipe temporary video cache immediately
                if os.path.exists(temp_name):
                    os.remove(temp_name)
                    
        elif trigger_analysis and uploaded_file is None:
            st.warning("Please supply a video asset track before invoking core analysis engines.")
        else:
            st.info("Awaiting structural parameters. Upload an active media asset stream to begin compilation.")


# =========================================================
# 2. THE AUTOMATED BROWSER INVOCATION BOOTSTRAPPER
# =========================================================
if __name__ == "__main__":
    from streamlit import runtime
    
    # Check if the file is already executing safely within the active Streamlit instance context
    if runtime.exists():
        run_ui_dashboard()
    else:
        # If run directly by standard 'python app.py', intercept arguments 
        # and boot up the Streamlit server completely automatically.
        from streamlit.web import cli as stcli
        print("[Bootstrapper] Direct python invocation captured. Launching core Streamlit UI...")
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())