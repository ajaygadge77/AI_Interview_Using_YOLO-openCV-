import streamlit as st
import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Interview Proctor", layout="wide")
st.title("🎥 Live Interview Co-Pilot")

model = YOLO("yolov8n.pt")  # Auto-downloads on first run

st.sidebar.header("🔒 Upload Student Reference Image")
uploaded_img = st.sidebar.file_uploader("Upload student's face image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    ref_image = face_recognition.load_image_file(uploaded_img)
    try:
        reference_encoding = face_recognition.face_encodings(ref_image)[0]
        st.sidebar.success("✅ Reference face encoded")
    except:
        st.sidebar.error("❌ Face not found in reference image.")
        reference_encoding = None
else:
    reference_encoding = None

if reference_encoding is not None and st.button("▶ Start Interview Monitoring"):

    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, face_locations)

        verified = any(face_recognition.compare_faces([reference_encoding], enc)[0] for enc in encodings)

        results = model(frame)[0]
        cheating = False

        person_count = sum(1 for box in results.boxes if int(box.cls[0]) == 0)
        if person_count > 1:
            cheating = True

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label in ["cell phone", "book", "laptop"]:
                cheating = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif label == "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Identity: {'✅ Verified' if verified else '❌ Not Verified'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Cheating: {'⚠️ Detected' if cheating else '✅ Clear'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
