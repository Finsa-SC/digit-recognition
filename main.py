from numpy import shape
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2
import joblib

# Load data
digits = load_digits()
x = digits.images
y = digits.target

print("Before reshape:", shape(x))  # (1797, 8, 8)

# Pipihkan x
n_samples = x.shape[0]
x = x.reshape((n_samples, -1))  # Jadi (1797, 64)

print("After reshape:", shape(x))  # (1797, 64)

# Split train-test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

print("Train shape:", shape(x_train), shape(y_train))
print("Test shape:", shape(x_test), shape(y_test))

# Buat model SVM
model = SVC()

model.fit(x_train, y_train)

# Tes akurasi
score = model.score(x_test, y_test)
print(f"Akurasi: {score * 100:.2f}%")

# Simpan model ke file
joblib.dump(model, "svm_digits_model.pkl")
print("Model disimpan sebagai 'svm_digits_model.pkl'")

# Load model
model = joblib.load("svm_digits_model.pkl")

# Buka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak bisa dibuka!")
    exit()

# Buat window fullscreen tanpa navbar
cv2.namedWindow("Digit Recognition", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Digit Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal ambil gambar!")
        break
    frame = cv2.flip(frame, 1)

    # Buat kotak tempat tulisan
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Ambil area dalam kotak
    roi = frame[y1:y2, x1:x2]

    # Preprocessing ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)        # Ke grayscale
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)  # Resize ke 8x8
    normalized = resized / 16.0                         # Sama kayak dataset digits
    flattened = normalized.flatten()                   # Jadi array [64]

    # Prediksi
    pred = model.predict([flattened])
    cv2.putText(frame, f"Prediksi: {pred[0]}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame
    cv2.imshow("Digit Recognition", frame)

    # Tekan 'q' buat keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()