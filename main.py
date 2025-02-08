import face_recognition as fr
import cv2
import numpy as np
from datetime import datetime

source = 'https://192.168.0.101:8080/video'
scale1 = 10
scale2 = 1/scale1

def face_confidence(face_distance, face_match=0.1):
    range: float = (1.0 - face_match)
    linear_val: float = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * pow((linear_val - 0.5) * 2, 0.2)) * 100)
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations: list = []
    face_encodings: list = []
    face_names: list = []
    know_face_encodings: list = []
    know_face_names: list = []

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        self.know_face_encodings = np.load('know_face_encodings.npy')
        self.know_face_names = np.load('know_face_names.npy')

    def run_recognition(self):
        vid = cv2.VideoCapture()
        vid.open(source)

        print(f'\u001b[38;5;226mVideo FPS: {vid.get(cv2.CAP_PROP_FPS)}')
        if not vid.isOpened():
            raise "Video not found"
        i = datetime.now()

        while True:
            ret, frame = vid.read()
            if int(vid.get(1)) % 25 == 0:
                print(f'Time from last frame: {datetime.now() - i}')
                i = datetime.now()
                small_frame = cv2.resize(frame, (0, 0), fx=scale2, fy=scale2)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                self.face_locations = fr.face_locations(rgb_small_frame)

                start = datetime.now()
                self.face_encodings = fr.face_encodings(rgb_small_frame, self.face_locations)
                print(f'\u001b[38;5;14m{self.face_encodings}\u001b[0m')
                print(f'\u001b[38;5;226mTime for encoding: {datetime.now() - start}')

                if self.face_encodings:
                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        matches: list = fr.compare_faces(self.know_face_encodings, face_encoding)
                        name: str = 'Unknown'
                        confidence: str = '(Unknown)'

                        face_distances = fr.face_distance(self.know_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = self.know_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                        self.face_names.append(f'{name} ({confidence})')
                        print(f'{name} ({confidence})')

                    for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                        top *= scale1
                        right *= scale1
                        bottom *= scale1
                        left *= scale1

                        color = (255, 0, 0) if name == 'Unknown' else (0, 255, 0)

                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(frame, (left, top - 35), (right, top), color, -1)
                        cv2.putText(frame, name, (left + 6, top-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                cv2.imshow('face', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        # vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    frec = FaceRecognition()
    frec.run_recognition()