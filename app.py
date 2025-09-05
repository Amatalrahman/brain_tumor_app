import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ----------------------------
# Load model & class names
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_mobilenetv2.h5")

@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    # Handle both list and dict formats
    if isinstance(class_names, dict):
        class_names = {int(k): v for k, v in class_names.items()}
    elif isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    return class_names

model = load_model()
class_names = load_class_names()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")
st.title("🧠 Brain Tumor Classification (MobileNetV2)")
st.write("Upload a brain MRI/CT image and the model will classify the tumor type.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a brain MRI/CT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    # Show result
    st.subheader("🔎 Prediction Result")
    st.write(f"**Class:** {class_names[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Confidence progress bar
    st.progress(confidence)

    # Show all class probabilities
    st.subheader("📊 Class Probabilities")
    prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
    st.json(prob_dict)
