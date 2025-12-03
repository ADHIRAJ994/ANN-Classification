import streamlit as st
import cv2
import numpy as np

st.title("OpenCV + Streamlit Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(image, channels="BGR")

    st.sidebar.header("Choose Image Processing Options")
    choice = st.sidebar.selectbox(
        "Select a filter",
        ("Grayscale", "Canny Edge Detection", "Blur", "Threshold", "Invert Colors")
    )

    if choice == "Grayscale":
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif choice == "Canny Edge Detection":
        result = cv2.Canny(image, 100, 200)

    elif choice == "Blur":
        result = cv2.GaussianBlur(image, (15, 15), 0)

    elif choice == "Threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    elif choice == "Invert Colors":
        result = cv2.bitwise_not(image)

    st.subheader("Processed Image")

    
    if len(result.shape) == 2:
        st.image(result)
    else:
        st.image(result, channels="BGR")
