import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import os
from streamlit_drawable_canvas import st_canvas

# Streamlit UI setup
st.title("Advanced Image Restoration with Inpainting 🎨")

# Upload Image
uploaded_file = st.file_uploader("Upload a damaged image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Ensure RGB mode
    image_np = np.array(image)

    # Image Preprocessing Options
    st.sidebar.subheader("🛠️ Image Preprocessing")

    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
    sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0)

    # Apply Enhancements
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)

    # Save temporarily for OpenCV processing
    temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    cv2.imwrite(temp_img_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    # Brush Options
    st.sidebar.subheader("🖌️ Brush Settings")
    stroke_width = st.sidebar.slider("Brush Size", 1, 30, 5)
    stroke_color = st.sidebar.color_picker("Pick Brush Color", "#FFFFFF")  # Default white

    # Convert HEX color to RGBA
    def hex_to_rgba(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)

    stroke_color_rgba = hex_to_rgba(stroke_color)

    # Streamlit Draw Canvas
    st.subheader("Draw over the damaged areas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",  # Transparent white
        stroke_width=stroke_width,
        stroke_color=stroke_color,  # Mask color
        background_image=image,
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="freedraw",
        key="canvas",
    )

    # Process mask when user submits
    if st.button("Restore Image"):
        if canvas_result.image_data is not None:
            # Convert drawn mask to grayscale
            mask = np.array(canvas_result.image_data, dtype=np.uint8)  # Ensure uint8 format
            mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)


            # Threshold the mask to ensure binary values (0 or 255)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # Perform inpainting
            restored_image = cv2.inpaint(np.array(image), mask, 3, cv2.INPAINT_TELEA)

            # Display original and restored images side by side
            st.subheader("Comparison: Original vs Restored")

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original Damaged Image", use_column_width=True)

            with col2:
                st.image(restored_image, caption="Restored Image", use_column_width=True)

            # Download option
            restored_filename = "restored_image.png"
            cv2.imwrite(restored_filename, cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))
            with open(restored_filename, "rb") as file:
                st.download_button("Download Restored Image", file, file_name="restored_image.png", mime="image/png")

    # Cleanup temp files
    os.remove(temp_img_path)  
