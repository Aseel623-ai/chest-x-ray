import streamlit as st
from PIL import Image
import os
from transformers import pipeline
import gdown


st.set_page_config(page_title="Chest X-ray Analyzer", layout="centered")
st.header("📷 Chest X-ray Classification")

@st.cache_resource
def load_model():
    files = {
        "config.json": {
            "file_id": "14M2rmv00uGCT7xbq7nHu7jkUaSsTQ5OG",
            "output": "config.json"
        },
        "model.safetensors": {
            "file_id": "1v90JJcPsad13gtxMluqCRau5HBmonjUH",
            "output": "model.safetensors"
        },
        "preprocessor_config.json": {
            "file_id": "1ycZG5YhATFS67-zODHZhLNY8WE7hphH9",
            "output": "preprocessor_config.json"
        }
    }

    model_dir = "chest_xray_model"
    os.makedirs(model_dir, exist_ok=True)

    try:
        for file_name, file_info in files.items():
            output_path = os.path.join(model_dir, file_name)

            if not os.path.exists(output_path):
                st.info(f"Downloading {file_name}...")
                gdown.download(
                    f"https://drive.google.com/uc?id={file_info['file_id']}",
                    output_path,
                    quiet=False
                )

            if not os.path.exists(output_path):
                st.error(f"❌ Failed to download: {file_name}")
                return None

        st.success("✅ Model files loaded successfully.")

        return pipeline(
            "image-classification",
            model=model_dir,
            device="cpu"
        )

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# تحميل النموذج
model = load_model()

if model is None:
    st.warning("Model couldn't be loaded.")
else:
    uploaded_file = st.file_uploader("Upload an X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing..."):
                    predictions = model(img)
                    top_prediction = predictions[0]
                    label = top_prediction['label']
                    score = top_prediction['score'] * 100

                    st.markdown(f"### 🩺 Diagnosis: **{label}**")
                    st.markdown(f"**Confidence:** {score:.2f}%")

        except Exception as e:
            st.error(f"❌ Error analyzing image: {str(e)}")