import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from collections import deque
import winsound
import time

classes = ["drowsy", "notdrowsy"]

prediction_history = deque(maxlen=3)
last_beep_time = 0

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]

        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face = frame[y1:y2, x1:x2]

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(face_rgb)
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        prediction_history.append(pred)
        stable_pred = max(set(prediction_history), key=prediction_history.count)
        label = classes[stable_pred]

        if label == "drowsy":
            color = (0, 0, 255)

            current_time = time.time()
            if current_time - last_beep_time > 1.2:
                winsound.Beep(1800, 200)
                winsound.Beep(1200, 200)
                winsound.Beep(1800, 200)
                last_beep_time = current_time
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"{label} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    else:
        prediction_history.clear()

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    cv2.imshow("Drowsy Driver Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()