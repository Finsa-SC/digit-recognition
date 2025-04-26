from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from numpy import shape
import joblib
import cv2

digits = load_digits()
x = digits.images
y = digits.target

print("Before reshape:", shape(x)) 

n_samples = x.shape[0]
x = x.reshape((n_samples, -1)) 

print("After reshape:", shape(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

print("Train shape:", shape(x_train), shape(y_train))
print("Test shape:", shape(x_test), shape(y_test))

model = SVC()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(f"Accuracy: {score * 100:.2f}%")

joblib.dump(model, "svm_digits_model.pkl")
print("The model is saved as 'svm_digits_model.ppl'")

model = joblib.load("svm_digits_model.pkl")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcams can't be opened!")
    exit()

cv2.namedWindow("Digit Recognition", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Digit Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to take picture!")
        break
    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)        
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA) 
    normalized = resized / 16.0                        
    flattened = normalized.flatten()                   

    pred = model.predict([flattened])
    cv2.putText(frame, f"Prediksi: {pred[0]}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
