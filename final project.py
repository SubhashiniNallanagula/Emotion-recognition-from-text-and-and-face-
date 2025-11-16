import os
import warnings
import cv2
import time
import torch
import numpy as np
from deepface import DeepFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading

# Suppress TensorFlow warnings etc.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# ========== Setup model and tokenizer ==========
model_path = "./robert-goemotions-model_10_epoch"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

goemotions_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

face_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

latest_frame = None
frame_lock = threading.Lock()
running = True

def webcam_thread():
    global latest_frame, running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Could not open webcam.")
        running = False
        return

    print(" Webcam started")

    while running:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break
        
        with frame_lock:
            latest_frame = frame.copy()

        cv2.imshow("Webcam - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting webcam capture...")
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

thread = threading.Thread(target=webcam_thread)
thread.start()

time.sleep(1)

print("\nWebcam started")

try:
    while running:
        text = input("\nEnter your sentence (or 'quit' to exit): ")
        if text.strip().lower() == 'quit':
            print("Exiting program...")
            running = False
            break

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            text_probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        dominant_text_idx = int(np.argmax(text_probs))
        dominant_text_emotion = goemotions_labels[dominant_text_idx]

        with frame_lock:
            frame_for_face = latest_frame.copy() if latest_frame is not None else None

        dominant_face_emotion = None
        face_one_hot = None

        if frame_for_face is not None:
            try:
                result = DeepFace.analyze(frame_for_face, actions=['emotion'], enforce_detection=False)
                emotions_dict = result[0]['emotion']
                dominant_emotion_raw = result[0]['dominant_emotion']

                print(f"Facial emotions detected: {emotions_dict}")

                dominant_face_emotion = dominant_emotion_raw.lower()

            except Exception as e:
                print(" Face not detected or analysis failed.", e)
        else:
            print(" No webcam frame available for facial emotion detection.")

        print("\n========== TEXT EMOTION ==========")
        print(f"Detected Text Emotion: {dominant_text_emotion}")

        print("\n========== FACIAL EXPRESSION ==========")
        if dominant_face_emotion and dominant_face_emotion in face_emotions:
            print(f"Detected Facial Emotion: {dominant_face_emotion}")
            face_one_hot = np.zeros(len(face_emotions))
            idx = face_emotions.index(dominant_face_emotion)
            face_one_hot[idx] = 1
        else:
            print("⚠ No facial emotion detected or recognized.")

        if face_one_hot is not None:
            fused_vector = np.concatenate([text_probs, face_one_hot])
            print("\n========== FUSED VECTOR (Text + Face) ==========")
            print(fused_vector)
        else:
            print("⚠ Fusion vector not created due to missing facial emotion.")

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
    running = False

thread.join()
print("Program terminated.")
