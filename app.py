"""
FusionINV Streamlit Demo
Interactive web interface for infrared and visible image fusion
"""
import streamlit as st
import torch
from PIL import Image
import io
import os
import tempfile

# Set page config
st.set_page_config(
    page_title="FusionINV - Image Fusion",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(device="cuda"):
    """Load FusionINV model (cached)."""
    try:
        from fusioninv import FusionINV
        
        # Check device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            dtype = torch.float32
            st.warning("CUDA not available, using CPU (slower)")
        else:
            dtype = torch.bfloat16
            
        model = FusionINV(device=device, dtype=dtype)
        return model, None
    except Exception as e:
        return None, str(e)


def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 FusionINV</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Training-Free Multimodal Image Fusion using Stable Diffusion</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        lambda_vis = st.slider(
            "Visible Cues Strength (λ)",
            min_value=0.0,
            max_value=0.3,
            value=0.08,
            step=0.01,
            help="Controls how much visible style is injected into IR features"
        )
        
        num_steps = st.slider(
            "Diffusion Steps (T)",
            min_value=20,
            max_value=100,
            value=80,
            step=10,
            help="Total number of diffusion steps"
        )
        
        T1 = st.slider(
            "IR Injection Cutoff (T1)",
            min_value=num_steps // 2,
            max_value=num_steps,
            value=min(70, num_steps),
            step=5,
            help="Step to stop injecting IR features"
        )
        
        T2 = st.slider(
            "VIS Refinement Cutoff (T2)",
            min_value=0,
            max_value=T1,
            value=min(40, T1),
            step=5,
            help="Step to stop injecting VIS features"
        )
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=15.0,
            value=7.5,
            step=0.5,
            help="Classifier-free guidance scale"
        )
        
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            help="Set for reproducible results"
        )
        
        st.divider()
        
        device = st.selectbox(
            "Device",
            options=["cuda", "cpu"],
            index=0 if torch.cuda.is_available() else 1
        )
        
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Visible Image")
        vis_file = st.file_uploader(
            "Upload visible image",
            type=["png", "jpg", "jpeg"],
            key="vis"
        )
        if vis_file:
            vis_image = Image.open(vis_file)
            st.image(vis_image, caption="Visible Image", use_container_width=True)
            
    with col2:
        st.subheader("🌡️ Infrared Image")
        ir_file = st.file_uploader(
            "Upload infrared image",
            type=["png", "jpg", "jpeg"],
            key="ir"
        )
        if ir_file:
            ir_image = Image.open(ir_file)
            st.image(ir_image, caption="Infrared Image", use_container_width=True)
    
    # Text prompt
    prompt = st.text_input(
        "Text Prompt (optional)",
        value="",
        placeholder="e.g., 'enhance the person in the scene'",
        help="Optional text guidance for fusion"
    )
    
    # Fusion button
    st.divider()
    
    if st.button("🚀 Fuse Images", type="primary", use_container_width=True):
        if vis_file and ir_file:
            with st.spinner("Loading model and processing images..."):
                # Load model
                model, error = load_model(device)
                
                if error:
                    st.error(f"Failed to load model: {error}")
                    st.info("Make sure you have installed all dependencies:\n```\npip install -r requirements.txt\n```")
                    return
                    
                # Set parameters
                model.set_params(
                    lambda_vis=lambda_vis,
                    num_steps=num_steps,
                    T1=T1,
                    T2=T2
                )
                
                # Save uploaded files temporarily
                with tempfile.TemporaryDirectory() as tmpdir:
                    vis_path = os.path.join(tmpdir, "vis.png")
                    ir_path = os.path.join(tmpdir, "ir.png")
                    
                    vis_image.save(vis_path)
                    ir_image.save(ir_path)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Inverting visible image...")
                    progress_bar.progress(25)
                    
                    try:
                        # Run fusion
                        fused_image = model.fuse(
                            vis_image_path=vis_path,
                            ir_image_path=ir_path,
                            prompt=prompt,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            verbose=False
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Fusion complete!")
                        
                        # Display result
                        st.divider()
                        st.subheader("✨ Fused Result")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(vis_image, caption="Visible", use_container_width=True)
                        with col2:
                            st.image(ir_image, caption="Infrared", use_container_width=True)
                        with col3:
                            st.image(fused_image, caption="Fused", use_container_width=True)
                        
                        # Download button
                        buf = io.BytesIO()
                        fused_image.save(buf, format="PNG")
                        st.download_button(
                            label="📥 Download Fused Image",
                            data=buf.getvalue(),
                            file_name="fused_image.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Fusion failed: {str(e)}")
                        st.exception(e)
        else:
            st.warning("Please upload both visible and infrared images")
    
    # Info section
    with st.expander("ℹ️ About FusionINV"):
        st.markdown("""
        **FusionINV** is a training-free method for infrared and visible image fusion 
        based on Stable Diffusion v1.5.
        
        ### Key Features
        - 🚀 **Training-Free**: Uses pre-trained Stable Diffusion directly
        - 🎨 **Visible-Style Output**: Produces images compatible with vision models
        - 💬 **Text-Interactive**: Supports text-guided fusion control
        
        ### Parameters Explained
        - **λ (Lambda)**: Controls visible style injection strength
        - **T1**: Steps for IR feature injection (early stage)
        - **T2**: Steps for VIS feature refinement (middle stage)
        - **Guidance Scale**: CFG scale for quality enhancement
        
        ### Citation
        ```bibtex
        @article{liang2025fusioninv,
          title={FusionINV: A Diffusion-Based Approach for Multimodal Image Fusion},
          author={Liang, Pengwei and Jiang, Junjun and Ma, Qing and others},
          journal={IEEE Transactions on Image Processing},
          year={2025}
        }
        ```
        """)


if __name__ == "__main__":
    main()
