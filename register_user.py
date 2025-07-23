# register_user.py
import cv2
import os
import joblib
from utils import extract_face, get_embedding

DATA_DIR = 'data'
EMB_PATH = os.path.join(DATA_DIR, 'embeddings.pkl')
LBL_PATH = os.path.join(DATA_DIR, 'labels.pkl')

# Load or initialize
if os.path.exists(EMB_PATH):
    embeddings = joblib.load(EMB_PATH)
    labels = joblib.load(LBL_PATH)
else:
    embeddings, labels = [], []

def save():
    joblib.dump(embeddings, EMB_PATH)
    joblib.dump(labels, LBL_PATH)

cap = cv2.VideoCapture(0)
name = input("Enter new user's name: ")
print("[INFO] Press 'c' to capture, 'q' to finish.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    face = extract_face(frame)
    if face is not None:
        cv2.imshow("Captured Face", face)

    cv2.imshow("Video Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and face is not None:
        emb = get_embedding(face)
        embeddings.append(emb)
        labels.append(name)
        print(f"[INFO] Sample added for {name}")
    elif key == ord('q'):
        break

save()
print(f"[INFO] Registration completed for {name}")
cap.release()
cv2.destroyAllWindows()
