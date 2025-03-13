import face_recognition as fr
import cv2
import numpy as np
from pathlib import Path

# Параметри відео та розпізнавання
VIDEO_SOURCE = 'ch01_20250222095355.mp4'  # Джерело відео
SCALE_FACTOR = 1  # Коефіцієнт зменшення зображення (1 - без змін)
FRAME_SKIP = 10  # Пропуск кадрів для обробки (щоб не перевантажувати процесор)
ROI = (800, 100, 500, 500)  # Область інтересу (ROI) у відео (x, y, width, height)
ENCODINGS_PATH = Path("data/know_face_encodings.npy")  # Шлях до збережених енкодингів облич
NAMES_PATH = Path("data/know_face_names.npy")  # Шлях до збережених імен

class FaceRecognition:
    def __init__(self):
        """Ініціалізація класу FaceRecognition, завантаження відомих облич."""
        self.known_face_encodings = []  # Масив векторів облич
        self.known_face_names = []  # Масив імен облич
        self.load_encodings()

    def load_encodings(self):
        """Завантаження збережених енкодингів імен та облич."""
        self.known_face_encodings = np.load(ENCODINGS_PATH, allow_pickle=True)
        self.known_face_names = np.load(NAMES_PATH, allow_pickle=True)

    def face_confidence(self, face_distance: float, face_match=0.6) -> str:
        """Обчислення рівня впевненості у збігу обличчя."""
        range_val = (1.0 - face_match)
        linear_val = (1.0 - face_distance) / (range_val * 2.0)

        if face_distance > face_match:
            return f"{round(linear_val * 100, 2)}%"
        else:
            adjusted_val = linear_val + ((1.0 - linear_val) * ((linear_val - 0.5) * 2) ** 0.2)
            return f"{round(adjusted_val * 100, 2)}%"

    def process_frame(self, frame):
        """Обробка кадру: зменшення розміру, перетворення в RGB, пошук облич."""
        small_frame = cv2.resize(frame, (0, 0), fx=1 / SCALE_FACTOR, fy=1 / SCALE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = fr.face_locations(rgb_small_frame)  # Визначення місця облич
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)  # Отримання векторів облич
        face_names = []

        for face_encoding in face_encodings:
            matches = fr.compare_faces(self.known_face_encodings, face_encoding)
            name, confidence = "Unknown", "(Unknown)"

            face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)  # Індекс найкращого збігу

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = self.face_confidence(face_distances[best_match_index])

            face_names.append(f"{name} ({confidence})")

        # Малювання рамок навколо облич і виведення імені
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = [int(coord * SCALE_FACTOR) for coord in (top, right, bottom, left)]
            color = (0, 0, 255) if name.startswith("Unknown") else (0, 255, 0)  # Червоний для невідомих, зелений для знайомих

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, top - 35), (right, top), color, -1)
            cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Масштабування кадру перед виведенням

        return frame

    def run_recognition(self):
        """Запуск відеопотоку та розпізнавання облич."""
        vid = cv2.VideoCapture(VIDEO_SOURCE)

        if not vid.isOpened():
            raise ValueError("Відео не знайдено або не може бути відкрито!")

        while True:
            ret, frame = vid.read()
            if not ret:
                break

            x, y, w, h = ROI  # Визначення області інтересу (ROI)
            frame = frame[y:y + h, x:x + w]  # Обрізка кадру за ROI

            if int(vid.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP == 0:
                frame = self.process_frame(frame)  # Обробка кадру
                cv2.imshow("Face Recognition", frame)  # Відображення кадру

            if cv2.waitKey(1) == ord("q"):  # Вихід при натисканні 'q'
                break

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = FaceRecognition()
    recognizer.run_recognition()
