import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.colors as mcolors
from keras import mixed_precision

import streamlit as st

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

TARGET_SIZE = (256, 256)
CLUSTERS_LABELS = {
    0: "Background",
    1: "Head Accesories",
    2: "Upper Body Clothing",
    3: "Lower Body Clothing",
    4: "Hands and Arms",
    5: "Legs and Feet",
    6: "Face"
}
SEMANTIC_CLASSES = len(CLUSTERS_LABELS)

def imshow(img):
    colors = plt.cm.tab20.colors[:len(CLUSTERS_LABELS)]
    cmap = mcolors.ListedColormap(colors)
    fig, ax = plt.subplots()
    img = np.argmax(img, axis=-1)
    ax.imshow(img, cmap=cmap)
    if CLUSTERS_LABELS:
        # Create legend
        handles = [
            plt.Line2D(
                [0], [0],
                color=cmap(idx),
                lw=4,
                label=label
            )
            for idx, label in CLUSTERS_LABELS.items()
        ]
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.05, 1), loc="upper left",
            borderaxespad=0.0,
            title="Clusters"
        )
    ax.axis("off")
    st.pyplot(fig)


st.set_page_config(page_title="Semantic Segmentation", layout="wide")
st.title("Semantic Segmentation")
# get file from user
uploaded_file = st.file_uploader("Choose an image...", type=['.jpeg', '.jpg', '.png', '.bmp', '.gif'])
# load the image
if uploaded_file is None:
    st.stop()
image = load_img(uploaded_file, target_size=TARGET_SIZE)
image = img_to_array(image)


image_col, prediction_col = st.columns([0.38, 0.62])

with image_col:
    st.image(image.astype("uint8"), caption="Uploaded Image", use_container_width=True)

# load the model
if not (model := st.session_state.get("model")):
    model = tf.keras.models.load_model("./data/models/deeplab_Adam_model.keras")
    st.session_state["model"] = model

# predict the image
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)[0]
with prediction_col:
    imshow(prediction)
