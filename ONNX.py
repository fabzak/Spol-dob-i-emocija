import cv2
import mediapipe as mp
import onnxruntime as ort
import os
import tkinter as tk
from tkinter import simpledialog
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def list_available_cameras():
    available_cameras = []
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

def select_camera_gui():
    cameras = list_available_cameras()
    if not cameras:
        print("Nisu pronađene kamere.")
        exit()
    
    root = tk.Tk()
    root.withdraw() 

    camera_options = "\n".join([f"{i}: Kamera {index}" for i, index in enumerate(cameras)])
    camera_selection_message = f"Odaberi kamera indeks:\n{camera_options}"
    selected_camera = simpledialog.askinteger("Kamera", camera_selection_message)
    
    if selected_camera is None or selected_camera not in cameras:
        print("Error u izboru.")
        exit()
    return selected_camera

gender_model_path = "models/gender_googlenet.onnx"
age_model_path = "models/age_googlenet.onnx"

try:
    gender_session = ort.InferenceSession(gender_model_path)
    age_session = ort.InferenceSession(age_model_path)
except Exception as e:
    print("Error ONNX model:", e)
    exit()

gender_list = ["Male", "Female"]
age_list = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", 
    "(38-43)", "(48-53)", "(60-100)"
]

camera_input = select_camera_gui()
video_capture = cv2.VideoCapture(camera_input)

def detect_faces_with_mediapipe(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append((x, y, w, h))
    return faces

def predict_age_gender(face_img):
    try:
        face_img_resized = cv2.resize(face_img, (224, 224))
        face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
        
        input_blob = np.expand_dims(face_img_rgb, axis=0).astype(np.float32) / 255.0
        input_blob = np.transpose(input_blob, (0, 3, 1, 2))  

        gender_preds = gender_session.run(None, {gender_session.get_inputs()[0].name: input_blob})[0]
        gender_index = np.argmax(gender_preds)
        gender = gender_list[gender_index]

        age_preds = age_session.run(None, {age_session.get_inputs()[0].name: input_blob})[0]
        age_index = np.argmax(age_preds)
        age = age_list[age_index]

        return gender, age
    except Exception as e:
        print("Error:", e)
        return "Unknown", "Unknown"

mode = 'RGB'

while True:
    result, video_frame = video_capture.read()
    if not result:
        break

    faces = detect_faces_with_mediapipe(video_frame)

    for (x, y, w, h) in faces:
        face_img = video_frame[y:y+h, x:x+w]

        gender, age = predict_age_gender(face_img)

        label = f"{gender}, {age}"
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            video_frame, label, (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    if mode == 'R':
        display_frame = video_frame.copy()
        display_frame[:, :, 1] = 0
        display_frame[:, :, 0] = 0
    elif mode == 'G':
        display_frame = video_frame.copy()
        display_frame[:, :, 2] = 0
        display_frame[:, :, 0] = 0
    elif mode == 'B':
        display_frame = video_frame.copy()
        display_frame[:, :, 1] = 0
        display_frame[:, :, 2] = 0
    else:
        display_frame = video_frame

    cv2.imshow("Detekcija lica i prepoznavanje s Mediapipe i ONNX modelom", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        mode = 'R'
    elif key == ord("g"):
        mode = 'G'
    elif key == ord("b"):
        mode = 'B'
    elif key == ord("f"):
        mode = 'BGR'

video_capture.release()
cv2.destroyAllWindows()
