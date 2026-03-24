import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Object Detector Pro", page_icon="🔍", layout="wide")

# --- 2. PROFESSIONAL DARK CSS (Compact Styling) ---
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.92)),
                          url('https://images.unsplash.com/photo-1518770660439-4636190af475');
        background-size: cover; background-attachment: fixed; color: white;
    }
    
    
    .stSelectbox, .stSlider, .stFileUploader {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px; padding: 10px;
    }

   
    div.stButton > button:first-child {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 20px !important;
        padding: 8px 30px !important;
        width: auto !important;
        display: block; 
        margin: 0 auto; /* Center the button */
        font-weight: bold !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOAD ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')
model = load_yolo()

# --- 4. HEADER ---
st.title("🔍 Advanced AI Object Detection")
st.write("Real-time inference using YOLOv8 optimized for professional deployment.")

# --- 5. COMPACT UI CONTROLS (Columns වලට බෙදා ඇත) ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    source = st.selectbox("Select Input Source", ["Image Upload", "Live Webcam"])
with col2:
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.45)

# --- 6. DETECTION LOGIC ---
if source == "Image Upload":
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop an image here (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        with st.spinner("Processing..."):
            results = model(img_array, conf=confidence)
            res_plotted = results[0].plot()
            
            st.divider()
            st.subheader("Inference Result")
            st.image(res_plotted, use_container_width=True)

elif source == "Live Webcam":
    st.info("Ensure your webcam is accessible.")
    if st.button("START WEBCAM STREAM"):
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        while True:
            ret, frame = camera.read()
            if not ret: break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame, conf=confidence)
            res_plotted = results[0].plot()
            FRAME_WINDOW.image(res_plotted)
            
        camera.release()