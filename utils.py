# utils.py
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

def extract_face(image, required_size=(160, 160)):
    results = detector.detect_faces(image)
    if results:
        x1, y1, w, h = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        face = image[y1:y1 + h, x1:x1 + w]
        face = cv2.resize(face, required_size)
        return face
    return None

def get_embedding(face_pixels):
    return embedder.embeddings([face_pixels])[0]
