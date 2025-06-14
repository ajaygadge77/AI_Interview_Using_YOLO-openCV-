# AI_Interview_Using_YOLO-openCV-
# 🎥 InterviewGuard — AI-Powered Interview Monitoring System

**InterviewGuard** is a real-time AI-driven proctoring system that verifies candidate identity and detects suspicious activity during online interviews. It combines face recognition with object detection and eye movement tracking to ensure interview integrity in academic or professional settings.

---

## 🔍 Key Features

- ✅ **Face Verification**  
  Matches the candidate's face against a pre-stored reference image using `face_recognition`.

- 👁️ **Eye Movement Detection**  
  Detects frequent eye deviations (left/right) to identify distraction or cheating attempts.

- 🎯 **Cheating Detection with YOLOv8**  
  Uses Ultralytics YOLOv8 to detect unauthorized objects like:
  - Mobile phones  
  - Books  
  - Laptops  
  - Multiple people in the frame

- 🖥️ **Live Monitoring Interface**  
  Built with Streamlit for real-time feedback and visual alerts.

---

## 🧰 Tech Stack

- Python 3.10
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Streamlit
- NumPy

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/interviewguard.git
cd interviewguard
2. Set up environment (Miniconda recommended)
bash
Copy
Edit
conda create -n interviewcopilot python=3.10 -y
conda activate interviewcopilot
conda install -c conda-forge dlib face_recognition
pip install streamlit opencv-python numpy ultralytics
3. Place student reference image
Create a folder named student_images/ and add a reference image named like:

bash
Copy
Edit
student_images/ajay_reference.jpg
4. Run the app
bash
Copy
Edit
streamlit run app.py
