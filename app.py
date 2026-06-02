import streamlit as st
import os
import sys

from video_handler import extract_key_frame
from cnn_encoder import CNNEncoder
from vlm_reasoner import VLMReasoner

def run_ui_dashboard():
    st.set_page_config(page_title="VigilAI Core Dashboard", page_icon="🛡️", layout="wide")

    st.title("🛡️ VigilAI: Dynamic Multi-Frame Analytics Portal")
    st.caption("Architecture: Modular pipelines executing over an interactive, deduplicated chronological sampling layout.")
    st.markdown("---")

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

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload Input Stream Assets")
        uploaded_file = st.file_uploader("Drop operational video feed here...", type=["mp4", "avi", "mov", "mkv"])
        
        st.markdown("### ⚙️ Time-Step Sampling Settings")
        
        # UPDATED: Max value expanded from 5 to 10 seconds
        interval_choice = st.slider(
            "Sample a keyframe every 'X' seconds:", 
            min_value=1, 
            max_value=10, 
            value=2,
            help="Setting to 10s creates a wide summary step across long video clips, skipping frozen duplication frames automatically."
        )
        
        custom_prompt = st.text_area("🧠 AI Query Directives", value="Is there any danger, threat, or operational issue in this video frame? Explain clearly.")
        trigger_analysis = st.button("🚀 Execute In-Memory Pipeline", use_container_width=True)

    with col2:
        st.subheader("📊 Pipeline Diagnostics Engine")
        
        if trigger_analysis and uploaded_file is not None:
            temp_name = f"ui_temp_cache_{uploaded_file.name}"
            with open(temp_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            status_bar = st.empty()
            progress = st.progress(0)
            
            try:
                # Execution Node 1: Pass selected interval seconds straight down
                status_bar.info(f"⚙️ Step 1/3: Extracting timestamped segments every {interval_choice}s...")
                progress.progress(25)
                pil_frame = extract_key_frame(temp_name, interval_seconds=interval_choice)
                
                st.image(pil_frame, caption=f"Module 1: Deduplicated {interval_choice}s Time-Interval Storyboard", use_container_width=True)
                
                # Execution Node 2: Run CNN embedding layer
                status_bar.info("⚙️ Step 2/3: Generating structural ResNet50 feature maps via Module 2...")
                progress.progress(60)
                structural_features = cnn_backbone.extract_features(pil_frame)
                
                # Execution Node 3: Generate text summary
                status_bar.info("⚙️ Step 3/3: Evaluating multimodal text token mappings via Module 3...")
                progress.progress(85)
                ai_report = vlm_engine.analyze_image_context(pil_frame, custom_prompt)
                
                progress.progress(100)
                status_bar.success("✅ Multi-Module pipeline successfully processed.")
                
                st.markdown("### 📝 Generated Incident Log Report")
                st.success(ai_report)
                
                with st.expander("🛠️ View Isolated Module Metas"):
                    st.json({
                        "Source Target Asset": uploaded_file.name,
                        "Time Step Interval Chosen": f"Every {interval_choice} seconds",
                        "Module 2 Tensor Layer Shape Output": list(structural_features.shape)
                    })
                    
            except Exception as err:
                status_bar.error(f"Critical execution block fault: {err}")
                
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
                    
        elif trigger_analysis and uploaded_file is None:
            st.warning("Please supply a video asset track before invoking core analysis engines.")
        else:
            st.info("Awaiting structural parameters. Upload an active media asset stream to begin compilation.")

if __name__ == "__main__":
    from streamlit import runtime
    if runtime.exists():
        run_ui_dashboard()
    else:
        from streamlit.web import cli as stcli
        print("[Bootstrapper] Direct python invocation captured. Launching core Streamlit UI...")
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())