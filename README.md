
Real-Time Face Recognition System (Hybrid Pipeline)
==================================================

A modular, real-time face recognition system using MTCNN for face detection, FaceNet for feature embedding, and cosine similarity for identity recognition. Supports real-time user registration and deletion without model retraining.

--------------------------------------------------
Project Structure
--------------------------------------------------

face_hybrid_pipeline/
├── data/                     # Saved embeddings and labels
│   ├── embeddings.pkl
│   └── labels.pkl
├── utils.py                  # Face detection and embedding logic
├── recognize.py              # Real-time recognition script
├── register_user.py          # Register new users
├── delete_user.py            # Delete registered users
├── requirements.txt          # Python dependencies
└── README.txt                # Documentation

--------------------------------------------------
Features
--------------------------------------------------

- Real-time webcam-based face recognition
- Cosine similarity matcher (no classifier retraining)
- Add/remove users on-the-fly
- Persistent data using joblib (.pkl files)
- Fast and accurate embeddings via keras-facenet

--------------------------------------------------
Requirements
--------------------------------------------------

Python Version: Python 3.7–3.11 recommended

Install Dependencies:

    pip install -r requirements.txt

requirements.txt:

    opencv-python
    mtcnn
    keras-facenet
    scikit-learn
    numpy
    joblib

--------------------------------------------------
Getting Started
--------------------------------------------------
Extract Files and Repo is ready to run. 

--------------------------------------------------
Full Pipeline Instructions
--------------------------------------------------

1. Register a New User

    python register_user.py

    - Enter the name when prompted.
    - Webcam will open.
    - Press:
        c to capture face image
        q to finish registration

2. Real-Time Face Recognition

    python recognize.py

    - Webcam will open.
    - If a face matches a known user, their name will be displayed.
    - Press q to quit.

3. Delete a Registered User

    python delete_user.py

    - Lists existing users.
    - Enter the name you want to delete.
    - Removes all associated embeddings.

--------------------------------------------------
How It Works
--------------------------------------------------

- Face Detection: MTCNN detects the face bounding box in each frame.
- Embedding Generation: FaceNet (via keras-facenet) converts face images into 128-dimensional vectors.
- Recognition: Uses cosine similarity between input and stored embeddings.

--------------------------------------------------
Notes
--------------------------------------------------

- You can add multiple samples per user during registration.
- Data is persistent across sessions via .pkl files.
- Accuracy improves with better lighting and more samples per person.

--------------------------------------------------
License
--------------------------------------------------

MIT License – free to use, modify, and distribute.
