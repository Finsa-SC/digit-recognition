# Digit Recognition with SVM

A real-time digit recognition application using a webcam with the Support Vector Machine (SVM) algorithm.

```Use Scikit-Learn Library for this Project```

## 🗒️ Description

This application uses a digit dataset from scikit-learn to train an SVM model that can recognize digits 0-9. Then, the application uses a webcam to apply digit recognition in real-time. You can show handwritten digits to the webcam, and the application will attempt to recognize them.

## ✨ Features

- Trains an SVM model to recognize digits 0-9
- Real-time image capture through the webcam
- Simple image preprocessing (grayscale, resize, normalization)
- Real-time digit recognition
- Display of a box on the screen to guide the placement of digits

## ⚠️ System Requirements

- Python 3.7 or higher
- Webcam/camera
- Python libraries (see requirements.txt)

## 🧰 Installation

1. Clone this repository or download the source code
   ```bash
   git clone https://github.com/Finsa-SC/digit-recognition.git
   cd digit-recognition
   ```

2. Install the required libraries
   ```bash
   pip install -r requirements.txt
   ```

## ❓ How to Use

1. Run the program
   ```bash
   python main.py
   ```

2. A window will open displaying the webcam feed
3. Write a digit on a white sheet of paper and show it within the displayed green box
4. The application will display the predicted digit on the screen
5. Press 'q' to exit the application

## ❔ How the Program Works

1. **Model Training**:
   - The program loads the digit dataset from scikit-learn
   - Images are converted into 1-dimensional arrays for processing
   - Data is split into training and testing sets
   - The SVM model is trained with the training data
   - The model is saved to a file for future use

2. **Real-time Recognition**:
   - The SVM model is loaded from the file
   - The webcam is activated and displays the video feed
   - A box is displayed as a guide for digit placement
   - The image within the box is processed to match the training dataset format
   - The model predicts the displayed digit
   - The prediction result is displayed on the screen

## ⏳ Further Development

Some ideas for project development:
- Implementation of more advanced image preprocessing
- Using convolutional neural networks (CNN) for better accuracy
- Adding the capability to recognize multiple digits simultaneously
- Enhancing the user interface

## 📑 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🫂 Contributors

Finsa Kusuma Putra

## 👑 Acknowledgments

- Dataset from scikit-learn
- Inspiration from various handwriting recognition projects