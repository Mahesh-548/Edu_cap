# Voice Command System

This project is a Python application that allows you to control various system operations using voice commands. It leverages machine learning techniques to recognize and classify voice inputs and uses automation libraries to execute corresponding actions on your computer.

## Features

- **Voice Recognition**: Captures voice input through the microphone and converts it to text.
- **Neural Network Classifier**: Utilizes MFCC features and a trained deep learning model to accurately classify voice commands.
- **Fuzzy Matching**: Employs fuzzy matching to determine the best matching command from a predefined list.
- **System Automation**: Executes system operations such as opening applications, adjusting volume, taking screenshots, and more.
- **Exit Command**: Allows you to safely exit the application by saying "exit program".

## Requirements

- Python 3.x
- NumPy
- Pandas
- TensorFlow
- Scikit-learn
- SpeechRecognition
- PyAutoGUI
- Other standard libraries (pickle, os, difflib)

## How to Use

1. **Prepare the Dataset**: Ensure that the `voice_command_dataset.csv` file is present in your working directory.
2. **Install Dependencies**: Install the required libraries using pip:
   ```
   pip install numpy pandas tensorflow scikit-learn SpeechRecognition PyAutoGUI
   ```
3. **Train the Model**: Run the provided script to preprocess the dataset, train the neural network, and save the best performing model along with the label encoder and scaler.
4. **Start the System**: Execute the script. The system will initialize and prompt you with "Listening..." for voice input.
5. **Execute Commands**: Speak any command from the predefined list. The system will process the input and execute the corresponding system action.
6. **Exit**: Say "exit program" to gracefully terminate the application.

## Additional Information

- **Data Preprocessing**: The project extracts MFCC features from audio samples and applies standardization to improve model performance.
- **Model Persistence**: The best performing model is saved as `voice_command_model.h5`, and both the label encoder and scaler are stored using pickle for future use.
- **Customization**: You can expand the command vocabulary and adjust system actions by modifying the actions dictionary in the script.

