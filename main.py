import face_recognition as fr
import cv2
import numpy as np
from pathlib import Path

from loguru import logger as log

# Параметри відео та розпізнавання
# VIDEO_SOURCE = Path("ip")  # Джерело відео
SCALE_FACTOR = 1  # Коефіцієнт зменшення зображення (1 - без змін)
FRAME_SKIP = 50  # Пропуск кадрів для обробки (щоб не перевантажувати процесор)
ENCODINGS_PATH = Path("data/know_face_encodings.npy")  # Шлях до збережених енкодингів облич
NAMES_PATH = Path("data/know_face_names.npy")  # Шлях до збережених імен

class FaceRecognition:
    def __init__(self):
        """Ініціалізація класу FaceRecognition, завантаження відомих облич."""
        self.known_face_encodings = []  # Масив векторів облич
        self.known_face_names = []  # Масив імен облич
        self.load_encodings()

    @log.catch
    def load_encodings(self):
        """Завантаження збережених енкодингів імен та облич."""
        self.known_face_encodings = np.load(ENCODINGS_PATH, allow_pickle=True)
        self.known_face_names = np.load(NAMES_PATH, allow_pickle=True)


    @log.catch
    def process_frame(self, frame):
        """Обробка кадру: зменшення розміру, перетворення в RGB, пошук облич."""
        small_frame = cv2.resize(frame, (0, 0), fx=1 / SCALE_FACTOR, fy=1 / SCALE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = fr.face_locations(rgb_small_frame)  # Визначення місця облич
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)  # Отримання векторів облич
        face_names = []

        for face_encoding in face_encodings:
            matches = fr.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = [int(coord * SCALE_FACTOR) for coord in (top, right, bottom, left)]
            color = (0, 0, 255) if name.startswith("Unknown") else (0, 255, 0)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, top - 35), (right, top), color, -1)
            cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        return frame

    @log.catch
    def run_recognition(self):
        """Запуск відеопотоку та розпізнавання облич."""
        vid = cv2.VideoCapture(0)

        if not vid.isOpened():
            raise ValueError("Відео не знайдено або не може бути відкрито!")

        while True:
            ret, frame = vid.read()
            if not ret:
                log.error("Відео не знайдено або не може бути відкрито!")
                break

            frame = self.process_frame(frame)  # Обробка кадру
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = FaceRecognition()
    recognizer.run_recognition()
