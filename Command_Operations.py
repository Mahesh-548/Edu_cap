import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import speech_recognition as sr
import pyautogui
import os
import difflib

# Load dataset
dataset_path = "voice_command_dataset.csv"
df = pd.read_csv(dataset_path)

# Encode labels
label_encoder = LabelEncoder()
df.iloc[:, -1] = label_encoder.fit_transform(df.iloc[:, -1])  # Encode command labels as numbers

# Split features and labels
X = df.iloc[:, :-1].values  # MFCC features
y = df.iloc[:, -1].values   # Encoded command labels

# Feature Scaling (Standardization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save label encoder and scaler for future use
with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Improved Model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile model with learning rate scheduling
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
MODEL_PATH = "voice_command_model.h5"
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max")
early_stopping = EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6)

# Train Model
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test), batch_size=32,
                    callbacks=[checkpoint, early_stopping, reduce_lr])

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# Load best model
model = load_model(MODEL_PATH)

# Function to recognize voice input
def recognize_voice():
    """Captures and processes voice command input."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError:
            print("Error with the speech recognition service")
            return None

# Find best match command
def get_best_match(command, command_list):
    """Finds the best match for a command using fuzzy matching."""
    match = difflib.get_close_matches(command, command_list, n=1, cutoff=0.6)
    return match[0] if match else None

# Execute system operations
def perform_action(command):
    """Executes system operations based on recognized commands."""
    actions = {
        "open browser": lambda: os.system("start chrome"),
        "shutdown": lambda: os.system("shutdown /s /t 1"),
        "increase volume": lambda: pyautogui.press("volumeup"),
        "decrease volume": lambda: pyautogui.press("volumedown"),
        "mute": lambda: pyautogui.press("volumemute"),
        "open notepad": lambda: os.system("notepad"),
        "open calculator": lambda: os.system("calc"),
        "lock screen": lambda: os.system("rundll32.exe user32.dll,LockWorkStation"),
        "open task manager": lambda: os.system("taskmgr"),
        "open command prompt": lambda: os.system("cmd"),
        "play music": lambda: os.system("start wmplayer"),
        "pause music": lambda: pyautogui.press("playpause"),
        "next track": lambda: pyautogui.press("nexttrack"),
        "previous track": lambda: pyautogui.press("prevtrack"),
        "take screenshot": lambda: pyautogui.screenshot().save("screenshot.png"),
        "maximize window": lambda: pyautogui.hotkey("win", "up"),
        "minimize window": lambda: pyautogui.hotkey("win", "down"),
        "close window": lambda: pyautogui.hotkey("alt", "f4"),
        "copy": lambda: pyautogui.hotkey("ctrl", "c"),
        "paste": lambda: pyautogui.hotkey("ctrl", "v"),
        "cut": lambda: pyautogui.hotkey("ctrl", "x"),
        "select all": lambda: pyautogui.hotkey("ctrl", "a"),
        "undo": lambda: pyautogui.hotkey("ctrl", "z"),
        "redo": lambda: pyautogui.hotkey("ctrl", "y"),
        "open settings": lambda: os.system("start ms-settings:"),
        "open file explorer": lambda: os.system("explorer"),
        "refresh desktop": lambda: pyautogui.hotkey("f5"),
        "open control panel": lambda: os.system("start control"),
        "log off": lambda: os.system("shutdown /l"),
        "exit program": lambda: exit_program()  # Safe exit option
    }

    best_match = get_best_match(command, actions.keys())

    if best_match:
        print(f"Executing: {best_match}")
        actions[best_match]()
    else:
        print("Command not recognized!")

# Function to exit program
def exit_program():
    """Gracefully exits the voice command loop."""
    print("Exiting voice command program...")
    exit()

# Real-time voice control
if __name__ == "__main__":
    print("Voice command system initialized. Say 'exit program' to quit.")
    while True:
        command = recognize_voice()
        if command:
            perform_action(command)
