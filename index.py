import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import speech_recognition as sr
import pyautogui
import os
import difflib
from tensorflow.keras.models import load_model

# Load saved model, label encoder, and scaler
MODEL_PATH = "voice_command_model.h5"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(LABEL_ENCODER_PATH, "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Voice command dictionary
command_operations = {
    "open browser": "Opens Google Chrome",
    "shutdown": "Shuts down the computer",
    "increase volume": "Increases system volume",
    "decrease volume": "Decreases system volume",
    "mute": "Mutes the volume",
    "open notepad": "Opens Notepad",
    "open calculator": "Opens Calculator",
    "lock screen": "Locks the computer",
    "open task manager": "Opens Task Manager",
    "open command prompt": "Opens CMD",
    "play music": "Starts Windows Media Player",
    "pause music": "Pauses/plays the media",
    "next track": "Plays the next track",
    "previous track": "Plays the previous track",
    "take screenshot": "Takes and saves a screenshot",
    "maximize window": "Maximizes the active window",
    "minimize window": "Minimizes the active window",
    "close window": "Closes the active window",
    "copy": "Copies selected text",
    "paste": "Pastes copied text",
    "cut": "Cuts selected text",
    "select all": "Selects all text",
    "undo": "Undoes the last action",
    "redo": "Redoes the last undone action",
    "open settings": "Opens Windows settings",
    "open file explorer": "Opens File Explorer",
    "refresh desktop": "Refreshes the desktop",
    "open control panel": "Opens the Control Panel",
    "log off": "Logs out of the current session",
    "exit program": "Stops the voice command system"
}

# Find best match for a command
def get_best_match(command):
    match = difflib.get_close_matches(command, command_operations.keys(), n=1, cutoff=0.6)
    return match[0] if match else None

# Execute system operations
def perform_action(command):
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
        "exit program": lambda: stop_listening()
    }

    best_match = get_best_match(command)
    if best_match:
        st.success(f"Executing: {best_match}")
        actions[best_match]()
    else:
        st.error("Command not recognized!")

# Voice recognition function
def recognize_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            st.success(f"Recognized: {command}")
            return command
        except sr.UnknownValueError:
            st.warning("Could not understand audio")
            return None
        except sr.RequestError:
            st.error("Error with the speech recognition service")
            return None

# Streamlit UI
st.title("🎙️ Voice Command System")
st.markdown("### Speak commands to control your system")

# Display command descriptions in boxes
st.subheader("📌 Available Commands:")
for command, desc in command_operations.items():
    with st.expander(f"🔹 {command.capitalize()}"):
        st.markdown(f"✅ **Operation:** {desc}")

# Start & Stop buttons
if "listening" not in st.session_state:
    st.session_state.listening = False

def start_listening():
    st.session_state.listening = True

def stop_listening():
    st.session_state.listening = False
    st.success("Voice Command System Stopped.")

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Voice Recognization"):
        start_listening()

with col2:
    if st.button("⏹ Stop"):
        stop_listening()

# Continuous listening loop
if st.session_state.listening:
    while True:
        command = recognize_voice()
        if command:
            perform_action(command)
