
# recognize.py (Multi-face Support)
import cv2
import os
import joblib
import numpy as np
from utils import extract_face, get_embedding
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = 'data'
EMB_PATH = os.path.join(DATA_DIR, 'embeddings.pkl')
LBL_PATH = os.path.join(DATA_DIR, 'labels.pkl')

# Load saved data
if os.path.exists(EMB_PATH):
    embeddings = joblib.load(EMB_PATH)
    labels = joblib.load(LBL_PATH)
else:
    embeddings, labels = [], []

# Detector
detector = MTCNN()

def recognize(embedding, threshold=0.5):
    if not embeddings:
        return "Unknown"
    sims = cosine_similarity([embedding], embeddings)[0]
    idx = np.argmax(sims)
    return labels[idx] if sims[idx] > threshold else "Unknown"

cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    for result in results:
        x1, y1, w, h = result['box']
        x1, y1 = abs(x1), abs(y1)
        face = frame[y1:y1+h, x1:x1+w]
        try:
            face_resized = cv2.resize(face, (160, 160))
            emb = get_embedding(face_resized)
            name = recognize(emb)
            # Draw bounding box and name
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except:
            continue

    cv2.imshow("Multi-Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
