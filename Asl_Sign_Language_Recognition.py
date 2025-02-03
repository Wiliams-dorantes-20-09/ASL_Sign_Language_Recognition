import cv2
import os
import mediapipe as mp
import time
import speech_recognition as sr
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def capture_images(output_dir="dataset", labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), num_images=300, delay=0.5):
    """Captures images when a hand sign is detected for each letter in the ASL alphabet."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    for label in labels:
        label_dir = os.path.join(output_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        print(f"Detecting sign for: {label}. Press 'q' to quit.")
        count = 0
        last_capture_time = time.time()
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Capture image with a time delay to prevent duplicate frames
                    if time.time() - last_capture_time >= delay:
                        img_path = os.path.join(label_dir, f"{count}.jpg")
                        cv2.imwrite(img_path, frame)
                        count += 1
                        last_capture_time = time.time()
                        print(f"Captured {count}/{num_images} for {label}")
            
            cv2.putText(frame, f"Letter: {label} - Captured: {count}/{num_images}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Capture Images", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
    
    cap.release()
    cv2.destroyAllWindows()

def preprocess_images(image_dir="dataset"):
    """Preprocess images by resizing, normalizing, and converting labels."""
    images = []
    labels = []
    
    for label in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))  # Resize to (64, 64) or any preferred size
                img = img / 255.0  # Normalize the image
                images.append(img)
                labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def train_model(images, labels, epochs=30):
    """Train a CNN model using the captured images."""
    # Label encoding
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded, num_classes=26)  # One-hot encoding
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
    
    # Data Augmentation setup
    datagen = ImageDataGenerator(
        rotation_range=30,  # Random rotations between -30 and 30 degrees
        width_shift_range=0.2,  # Random horizontal shift
        height_shift_range=0.2,  # Random vertical shift
        shear_range=0.2,  # Random shear
        zoom_range=0.2,  # Random zoom
        horizontal_flip=True,  # Random horizontal flip
        fill_mode='nearest'  # Fill pixels after transformation
    )
    
    datagen.fit(X_train)
    
    # Model definition
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(26, activation='softmax')  # 26 letters in ASL alphabet
    ])
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Training the model with data augmentation
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_test, y_test))
    
    return model, label_encoder

def capture_audio_to_text(output_file="speech_text.txt"):
    """Captures audio and converts it to text, saving it in a plain text file."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for speech...")
        
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            
            with open(output_file, "a") as file:
                file.write(text + "\n")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError:
            print("Error with the speech recognition service")

if __name__ == "__main__":
    # Step 1: Capture images for the ASL alphabet
    capture_images()
    
    # Step 2: Preprocess captured images
    images, labels = preprocess_images()
    
    # Step 3: Train the model with the preprocessed images (including augmentation)
    model, label_encoder = train_model(images, labels, epochs=30)
    
    # Step 4: Optionally, save the trained model
    model.save("asl_sign_language_model.h5")
    print("Model trained and saved.")
    
    # Step 5: Capture audio and convert it to text
    capture_audio_to_text()
