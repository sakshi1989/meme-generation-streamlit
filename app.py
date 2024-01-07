import streamlit as st
from PIL import Image
from lib.util import generate_meme


# Set the title and description of the app
st.title("Meme Generation")

st.subheader("Select a model variation to use:")
model_name = st.selectbox("Model", ("InceptionV3", "ResNet50"))

if model_name:
    st.subheader("Upload an image below (jpg, jpeg, png), to generate a meme:")

    # Create a file uploader widget
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Generate a meme using the Finalized Model and the uploaded image
    if uploaded_image is not None:
        output = generate_meme(Image.open(uploaded_image), model_name)

        st.write("Uploaded Image:")

        _, col, _ = st.columns([1, 3, 1])

        with col:
            st.image(output, caption="Uploaded Image", use_column_width=True)
