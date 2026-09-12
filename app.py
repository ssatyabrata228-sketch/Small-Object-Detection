# ---------------- IMPORTS ----------------
import sys
import os

# ---------------- PROJECT PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFINET_DIR = os.path.join(BASE_DIR, "cfinet")
CFINET_CONFIG = os.path.join(
    CFINET_DIR,
    "configs",
    "cfinet",
    "faster_rcnn_r50_fpn_cfinet_1x.py"
)
CFINET_CHECKPOINT = os.path.join(
    BASE_DIR,
    "models",
    "cfinet",
    "latest.pth"
)

# Allow Python to find the local CFINet/MMDetection code
sys.path.insert(0, CFINET_DIR)

import pandas as pd

import json
import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image

from ultralytics import YOLO
from mmdet.apis import init_detector, inference_detector

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="VisionAI | Small Object Detection",
    
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "cfinet_model" not in st.session_state:
    st.session_state.cfinet_model = None

os.makedirs("work_dirs/roi_feats/cfinet", exist_ok=True)

# ---------------- IOU ----------------
def compute_iou(box1, box2):
    x1=max(box1[0],box2[0]); y1=max(box1[1],box2[1])
    x2=min(box1[2],box2[2]); y2=min(box1[3],box2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(box1[2]-box1[0])*(box1[3]-box1[1])
    a2=(box2[2]-box2[0])*(box2[3]-box2[1])
    union=a1+a2-inter
    return inter/union if union else 0

def filter_new(old_boxes, new_boxes):
    out=[]
    for n in new_boxes:
        keep=True
        for o in old_boxes:
            if compute_iou(o["bbox"],n["bbox"])>0.5:
                keep=False
                break
        if keep:
            out.append(n)
    return out
# ---------------- HYBRID MERGE ----------------
def hybrid_merge(boxes1, boxes2):
            final = []
            duplicates = []

            for box in boxes1 + boxes2:
                keep = True
                for f in final:
                    if compute_iou(box["bbox"], f["bbox"]) > 0.5:
                        if box["confidence"] <= f["confidence"]:
                            duplicates.append(box)
                            keep = False
                            break
                        else:
                            duplicates.append(f)
                            final.remove(f)
                            break

                if keep:
                    final.append(box)

            return final, duplicates   # ✅ FIXED

# ---------------- LOAD MODELS ----------------
def load_cfinet():
    if st.session_state.cfinet_model is None:

        if not os.path.exists(CFINET_CONFIG):
            st.error(f"CFINet config not found: {CFINET_CONFIG}")
            st.stop()

        if not os.path.exists(CFINET_CHECKPOINT):
            st.error(f"CFINet checkpoint not found: {CFINET_CHECKPOINT}")
            st.stop()

        st.session_state.cfinet_model = init_detector(
            CFINET_CONFIG,
            CFINET_CHECKPOINT,
            device="cuda:0"
        )
# ---------------- LOGIN ----------------
def login_page():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(255,255,255,0.6), rgba(255,255,255,0.7)),
                    url(https://images.unsplash.com/photo-1485827404703-89b55fcc595e);
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        animation: zoomBg 15s infinite alternate;
    }

    @keyframes zoomBg {
        0% { background-size: 100%; }
        100% { background-size: 110%; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Center using columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        # Center wrapper
        st.markdown("""
        <div style="
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
        ">
        """, unsafe_allow_html=True)

        # Title Box
        st.markdown("""
        <div style="
            background:linear-gradient(135deg, #2563EB, #1E3A8A);
            padding:25px;
            border-radius:20px;
            text-align:center;
            font-size:34px;
            font-weight:700;
            color:white;
            width:100%;
            margin-bottom:30px;
            box-shadow:0 10px 30px rgba(0,0,0,0.2);
        ">
         Small Object Detection
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='text-align:center;'>🔐 Secure Login</h3>", unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Login"):
            correct_username = os.getenv("VISIONAI_USERNAME", "admin")
            correct_password = os.getenv("VISIONAI_PASSWORD", "change-me")

            if username == correct_username and password == correct_password:
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)
# ---------------- FORCE WHITE BACKGROUND ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color:Light Gray;
        color: #3F5EFB;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------- FORCE FULL BLACK TEXT ----------------
st.markdown("""
<style>

/* 🌈 MAIN BACKGROUND (YOUR GRADIENT) */
.stApp {
    background: radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 81%);
}

/* Remove default padding */
.block-container {
    padding-top: 2rem;
}

/* Input fields */
input {
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* Labels */
label {
    color: white !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563EB, #1E3A8A);
    color: white !important;
    border-radius: 30px;
    padding: 14px;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

/* Button hover */
.stButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #1E40AF);
}

/* Checkbox */
.stCheckbox label {
    color: white !important;
}

/* Text */
body, p, span {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# ---------------- DETECTION PAGE ----------------
def detection_page():

    st.title("Object Detection")
    

    threshold = st.slider("Confidence Threshold",0.0,1.0,0.5)

    # TWO MODELS
    model_options = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "yolov5m.pt": "yolov5m.pt",
    "YOLO11n": "yolo11n.pt",
    "CFINet": "CFINet"
}


    model1_name = st.selectbox("Select Model 1", list(model_options.keys()))
    model2_name = st.selectbox("Select Model 2", list(model_options.keys()))

    model1 = model_options[model1_name]
    model2 = model_options[model2_name]
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file is None:
        st.warning("Upload image first")
        st.stop()

    image = Image.open(file).convert("RGB")
    st.image(image)

    if st.button("Run Detection"):

        img = np.array(image)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ---------------- MODEL 1 ----------------
        json_A = []

        if model1 != "CFINet":
            m1 = YOLO(model1)
            res1 = m1(img, conf=threshold)
            img1 = res1[0].plot()

            for b in res1[0].boxes:
                x1,y1,x2,y2 = map(int,b.xyxy[0])
                json_A.append({
                    "class": m1.names[int(b.cls[0])],
                    "confidence": float(b.conf[0]),
                    "bbox": [x1,y1,x2,y2]
                })

        else:
            load_cfinet()
            m1 = st.session_state.cfinet_model
            res1 = inference_detector(m1, img_cv)
            img1 = img_cv.copy()

            for i,b in enumerate(res1):
                if b is not None:
                    for bb in b:
                        if bb[4] < threshold:
                            continue
                        x1,y1,x2,y2 = map(int, bb[:4])
                        json_A.append({
                            "class": m1.CLASSES[i],
                            "confidence": float(bb[4]),
                            "bbox": [x1,y1,x2,y2]
                        })
                        cv2.rectangle(img1,(x1,y1),(x2,y2),(0,255,0),2)

        st.subheader("Model 1 Output")
        st.image(img1)
        st.json(json_A)
        if json_A:
            df_A = pd.DataFrame([
                {"Model": model1_name, "Class": obj["class"], "Confidence": obj["confidence"]}
                for obj in json_A
            ])
            st.subheader("Model 1 Table")
            st.dataframe(df_A)

        # ---------------- MODEL 2 ----------------
        json_B = []

        if model2 != "CFINet":
            m2 = YOLO(model2)
            res2 = m2(img, conf=threshold)

            for b in res2[0].boxes:
                x1,y1,x2,y2 = map(int,b.xyxy[0])
                json_B.append({
                    "class": m2.names[int(b.cls[0])],
                    "confidence": float(b.conf[0]),
                    "bbox": [x1,y1,x2,y2]
                })

        else:
            load_cfinet()
            m2 = st.session_state.cfinet_model
            res2 = inference_detector(m2, img_cv)

            for i,b in enumerate(res2):
                if b is not None:
                    for bb in b:
                        if bb[4] < threshold:
                            continue
                        x1,y1,x2,y2 = map(int, bb[:4])
                        json_B.append({
                            "class": m2.CLASSES[i],
                            "confidence": float(bb[4]),
                            "bbox": [x1,y1,x2,y2]
                        })

        # 🔥 FILTER (IMPORTANT)
        json_B_filtered = filter_new(json_A, json_B)

        # 🔥 DRAW ONLY NEW OBJECTS
        img2 = img.copy()
        for obj in json_B_filtered:
            x1,y1,x2,y2 = obj["bbox"]
            cv2.rectangle(img2,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(img2,f'{obj["class"]}:{obj["confidence"]:.2f}',
                        (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

        st.subheader("Model 2 Output (Only NEW Objects)")
        st.image(img2)
        st.json(json_B_filtered)
        if json_B_filtered:
            df_B = pd.DataFrame([
                {"Model": model2_name, "Class": obj["class"], "Confidence": obj["confidence"]}
                for obj in json_B_filtered
            ])
            st.subheader("Model 2 Table")
            st.dataframe(df_B)

        # ---------------- FINAL MERGE ----------------
        final_json = json_A + json_B_filtered

        st.subheader("Final Merged Output")
        st.json(final_json)
        max_len = max(len(json_A), len(json_B_filtered))  # for detection page


        data = []

        for i in range(max_len):
            row = {}

        # Model 1
            if i < len(json_A):
                row["Model 1"] = model1_name
                row["Class 1"] = json_A[i]["class"]
                row["Conf 1"] = round(json_A[i]["confidence"], 3)
            else:
                row["Model 1"] = ""
                row["Class 1"] = ""
                row["Conf 1"] = ""

        # Model 2
            if i < len(json_B_filtered):   # 👉 in hybrid use json_B
                row["Model 2"] = model2_name
                row["Class 2"] = json_B_filtered[i]["class"]
                row["Conf 2"] = round(json_B_filtered[i]["confidence"], 3)
            else:
                row["Model 2"] = ""
                row["Class 2"] = ""
                row["Conf 2"] = ""

            data.append(row)

        df_compare = pd.DataFrame(data)

        st.subheader("Model Comparison Table")
        st.dataframe(df_compare)
# ---------------- HYBRID PAGE ----------------
def hybrid_page():

    st.title(" Hybrid Detection (Advanced)")

    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    model_options = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "yolov5m.pt": "yolov5m.pt",
    "YOLO11n": "yolo11n.pt",
    "CFINet": "CFINet"
}

    model1_name = st.selectbox("Select Model A", list(model_options.keys()))
    model2_name = st.selectbox("Select Model B", list(model_options.keys()))

    model1 = model_options[model1_name]
    model2 = model_options[model2_name]

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file is None:
        st.warning("Upload image first")
        st.stop()

    image = Image.open(file).convert("RGB")
    st.image(image)

    if st.button("Run Hybrid Detection"):

        img = np.array(image)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        json_A, json_B = [], []

        # -------- MODEL A --------
        if model1 != "CFINet":
            m1 = YOLO(model1)
            res1 = m1(img, conf=threshold)

            for b in res1[0].boxes:
                x1,y1,x2,y2 = map(int,b.xyxy[0])
                json_A.append({
                    "class": m1.names[int(b.cls[0])],
                    "confidence": float(b.conf[0]),
                    "bbox": [x1,y1,x2,y2]
                })
        else:
            load_cfinet()
            m1 = st.session_state.cfinet_model
            res1 = inference_detector(m1, img_cv)

            for i,b in enumerate(res1):
                if b is not None:
                    for bb in b:
                        if bb[4] < threshold: continue
                        x1,y1,x2,y2 = map(int, bb[:4])
                        json_A.append({
                            "class": m1.CLASSES[i],
                            "confidence": float(bb[4]),
                            "bbox": [x1,y1,x2,y2]
                        })

        # -------- MODEL B --------
        if model2 != "CFINet":
            m2 = YOLO(model2)
            res2 = m2(img, conf=threshold)

            for b in res2[0].boxes:
                x1,y1,x2,y2 = map(int,b.xyxy[0])
                json_B.append({
                    "class": m2.names[int(b.cls[0])],
                    "confidence": float(b.conf[0]),
                    "bbox": [x1,y1,x2,y2]
                })
        else:
            load_cfinet()
            m2 = st.session_state.cfinet_model
            res2 = inference_detector(m2, img_cv)

            for i,b in enumerate(res2):
                if b is not None:
                    for bb in b:
                        if bb[4] < threshold: continue
                        x1,y1,x2,y2 = map(int, bb[:4])
                        json_B.append({
                            "class": m2.CLASSES[i],
                            "confidence": float(bb[4]),
                            "bbox": [x1,y1,x2,y2]
                        })

        # -------- HYBRID MERGE --------
        final_json,duplicates = hybrid_merge(json_A, json_B)
        

        # -------- DRAW FINAL --------
        img_final = img.copy()

        for obj in final_json:
            x1,y1,x2,y2 = obj["bbox"]
            cv2.rectangle(img_final,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(img_final,
                        f'{obj["class"]}:{obj["confidence"]:.2f}',
                        (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

        st.subheader("Hybrid Output")
        st.image(img_final)
        st.json(final_json)
        all_A = json_A
        all_B = json_B

        data = []

        max_len = max(len(all_A), len(all_B))

        for i in range(max_len):
            row = {}

        # -------- MODEL 1 --------
            if i < len(all_A):
                obj = all_A[i]
                status = "Removed" if obj in duplicates else "Kept"

                row["Model 1"] = model1_name
                row["Class 1"] = obj["class"]
                row["Conf 1"] = round(obj["confidence"], 3)
                row["Status 1"] = status
            else:
                row["Model 1"] = ""
                row["Class 1"] = ""
                row["Conf 1"] = ""
                row["Status 1"] = ""

        # -------- MODEL 2 --------
            if i < len(all_B):
                obj = all_B[i]
                status = "Removed" if obj in duplicates else "Kept"

                row["Model 2"] = model2_name
                row["Class 2"] = obj["class"]
                row["Conf 2"] = round(obj["confidence"], 3)
                row["Status 2"] = status
            else:
                row["Model 2"] = ""
                row["Class 2"] = ""
                row["Conf 2"] = ""
                row["Status 2"] = ""

            data.append(row)

        df_compare = pd.DataFrame(data)

        st.subheader("Hybrid Comparison Table (with Duplicate Info)")
        st.dataframe(df_compare)
# ---------------- MAIN ----------------
if st.session_state.logged_in:

    st.sidebar.title(" Navigation")

    page = st.sidebar.selectbox(
        "Go to",
        ["Detection", "Hybrid Detection"]
    )

    # 👉 PUSH LOGOUT TO BOTTOM
    st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # 👉 PAGE ROUTING
    if page == "Detection":
        detection_page()
    elif page == "Hybrid Detection":
        hybrid_page()

else:
    login_page()
