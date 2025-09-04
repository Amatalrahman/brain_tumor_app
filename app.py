import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "brisc_mobilenetv2_finetuned.keras"
    )  # Path to trained model
    return model

model = load_model()

# ---------------- Load Class Names ----------------
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

IMG_SIZE = (224, 224)

st.set_page_config(page_title="Brain Tumor Classification", layout="centered")

# ---------------- UI ----------------
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image and let the model classify it into one of the four categories.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Preprocess the image
    img_array = image.resize(IMG_SIZE)
    img_array = np.array(img_array, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: [1, 224, 224, 3]

    # 🔹 Try with normalization (even if model has Rescaling layer, just to debug)
    # img_array = img_array / 255.0  

    st.write("Image shape:", img_array.shape, "Min:", img_array.min(), "Max:", img_array.max())

    # Make prediction
    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[str(predicted_index)]

    # Debug print
    st.subheader("🛠 Debug Info")
    st.write("Raw predictions:", predictions.tolist())

    # Show prediction result
    st.subheader("🔍 Prediction Result")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Visualization (Bar Chart)
    st.subheader("📊 Confidence per Class")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES.values(), predictions[0], color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)
