import face_recognition as fr
import sys
import cv2
from datetime import datetime

source = 'https://192.168.0.102:8080/video'

scale1 = 0.1
scale2 = 10


class FaceRecognition:
    face_locations: list = []

    def run_recognition(self):
        vid = cv2.VideoCapture('0')
        vid.open(source)

        print(f'\u001b[38;5;226mVideo FPS: {vid.get(cv2.CAP_PROP_FPS)}')
        if not vid.isOpened():
            sys.exit('Video source not found...')
        i = datetime.now()

        while True:
            ret, frame = vid.read()
            if int(vid.get(1)) % 1 == 0:
                print(f'Time from last frame: {datetime.now() - i}')
                i = datetime.now()
                small_frame = cv2.resize(frame, (0, 0), fx=scale1, fy=scale1)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = fr.face_locations(rgb_small_frame)

                for top, right, bottom, left in self.face_locations:
                    top *= scale2
                    right *= scale2
                    bottom *= scale2
                    left *= scale2

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, top - 35), (right, top), (0, 0, 255), -1)
                    cv2.putText(frame, 'name', (left + 6, top-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('face', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    frec = FaceRecognition()
    frec.run_recognition()
