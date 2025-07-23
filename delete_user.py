# delete_user.py
import os
import joblib

DATA_DIR = 'data'
EMB_PATH = os.path.join(DATA_DIR, 'embeddings.pkl')
LBL_PATH = os.path.join(DATA_DIR, 'labels.pkl')

if os.path.exists(EMB_PATH):
    embeddings = joblib.load(EMB_PATH)
    labels = joblib.load(LBL_PATH)
else:
    print("[ERROR] No user data found.")
    exit()

print("Registered users:", sorted(set(labels)))
name = input("Enter name to delete: ")

if name not in labels:
    print("[WARN] User not found.")
    exit()

# Remove entries
new_embeds = [e for i, e in enumerate(embeddings) if labels[i] != name]
new_labels = [l for l in labels if l != name]

joblib.dump(new_embeds, EMB_PATH)
joblib.dump(new_labels, LBL_PATH)
print(f"[INFO] Deleted all entries for {name}")