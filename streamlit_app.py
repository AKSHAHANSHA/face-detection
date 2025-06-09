import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# Attendance folder and file
IMAGE_FOLDER = "static/IMAGE_FILES"
ATTENDANCE_FILE = "attendance.csv"

# Load known faces
@st.cache_data
def load_known_faces(folder):
    encodings = []
    names = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(img)[0]
        encodings.append(encoding)
        names.append(os.path.splitext(file)[0])
    return encodings, names

# Save attendance
def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        open(ATTENDANCE_FILE, "w").close()

    with open(ATTENDANCE_FILE, "r+") as f:
        lines = f.readlines()
        names = [line.split(",")[0] for line in lines]
        if name not in names:
            now = datetime.now().strftime("%H:%M:%S")
            f.write(f"{name},{now}\n")

# UI
st.title("👨‍🏫 Face Recognition Attendance System")
st.markdown("Click the button below to start webcam.")

if st.button("Start Camera"):
    encodings, names = load_known_faces(IMAGE_FOLDER)
    stframe = st.empty()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb)
        encodes = face_recognition.face_encodings(rgb, locations)

        for encode, loc in zip(encodes, locations):
            matches = face_recognition.compare_faces(encodings, encode)
            dist = face_recognition.face_distance(encodings, encode)
            match_index = np.argmin(dist)

            if matches[match_index]:
                name = names[match_index].upper()
                mark_attendance(name)
                y1, x2, y2, x1 = [v * 4 for v in loc]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
