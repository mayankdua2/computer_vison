import streamlit as st
from PIL import Image, ImageOps
import albumentations as A
import numpy as np
import io
import zipfile

st.markdown('''# **Image Augmentation App**
Choose the Transformations to Apply:.
''')

st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4A90E2;
            text-align: center;
        }
        .checkbox-label {
            font-size: 18px;
        }
        .uploader {
            border: 2px dashed #4A90E2;
            padding: 20px;
            text-align: center;
            font-size: 18px;
            color: #4A90E2;
        }
        .button {
            display: block;
            width: 100%;
            background-color: #4A90E2;
            color: white;
            font-size: 20px;
            border: none;
            padding: 15px;
            text-align: center;
            cursor: pointer;
        }
        .button:hover {
            background-color: #357ABD;
        }
        nav.navbar {
            margin: 0 auto;
            max-width: 80%;
        }
    </style>
""", unsafe_allow_html=True)

rotate = st.checkbox("Rotate", value=False, help="Rotate the image randomly within a specified range.")
horizontal_flip = st.checkbox("Horizontal Flip", value=False, help="Tilt the image horizontally.")
vertical_flip = st.checkbox("Vertical Flip", value=False, help="Flip the image vertically.")
brightness_contrast = st.checkbox("Random Brightness/Contrast", value=False, help="Apply random brightness and contrast adjustments.")
zoom = st.checkbox("Ultra Zoom", value=False, help="Zoom into the image very closely.")
black_and_white = st.checkbox("Black and White", value=False, help="Convert the image to black and white.")




augmentation_list = []
if rotate:
    augmentation_list.append(A.Rotate(limit=90, p=1.0))  # Always apply rotation up to 90 degrees
if horizontal_flip:
    augmentation_list.append(A.HorizontalFlip(p=0.5))  # Tilt the image horizontally
if vertical_flip:
    augmentation_list.append(A.VerticalFlip(p=1.0))  # Always apply vertical flip
if brightness_contrast:
    augmentation_list.append(A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0))  # Maximum brightness/contrast adjustment
if zoom:
    augmentation_list.append(A.RandomCrop(height=512, width=512, p=1.0))  # Random crop for ultra zoom
if black_and_white:
    augmentation_list.append(A.ToGray(p=1.0))  # Convert image to black and white

augmentation_pipeline = A.Compose(augmentation_list)


def augment_image(image, augmentation):
    image_np = np.array(image)
    augmented = augmentation(image=image_np)['image']
    return Image.fromarray(augmented)

uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            
            # Augment the image with selected transformations
            for transformation in augmentation_list:
                augmented_image = augment_image(image, transformation)
                
                # Save the augmented image to a BytesIO object
                img_bytes = io.BytesIO()
                augmented_image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                # Add the image to the zip file
                # Use the transformation name as part of the file name
                transformation_name = transformation.__class__.__name__.lower()
                zf.writestr(f"{transformation_name}_{uploaded_file.name}", img_bytes.read())

    zip_buffer.seek(0)


    st.download_button(
        label="Download Augmented Images as ZIP",
        data=zip_buffer,
        file_name="augmented_images.zip",
        mime="application/zip",
        key="download-button",
        help="Download all augmented images as a zip file."
    )
