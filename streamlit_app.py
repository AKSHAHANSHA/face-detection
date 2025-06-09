import streamlit as st
import face_recognition
import numpy as np
import cv2
import os

st.title("Face Recognition App")

image_files_dir = 'static/IMAGE_FILES'

# Load known faces
known_faces = []
known_names = []

for file in os.listdir(image_files_dir):
    if file.lower().endswith(('png', 'jpg', 'jpeg')):
        img_path = os.path.join(image_files_dir, file)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(file.split('.')[0])

uploaded_file = st.file_uploader("Upload a picture for recognition", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    locations = face_recognition.face_locations(img_rgb)
    encodings = face_recognition.face_encodings(img_rgb, locations)

    for (top, right, bottom, left), face_encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if matches and len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img_rgb, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(img_rgb, caption="Face Recognition Result", use_column_width=True)
