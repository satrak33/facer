import numpy as np
import os
from datetime import datetime
import face_recognition as fr


def save_faces():
    know_face_encodings: list = []
    know_face_names: list = []

    for img in os.listdir('../faces'):
        # start = datetime.now()

        face_img = fr.load_image_file(f'faces/{img}')
        face_encoding = fr.face_encodings(face_img)[0]

        know_face_encodings.append(face_encoding)
        know_face_names.append(img)

        # end = datetime.now()
        # print(f'\u001b[38;5;15mTime for img "{img}": {end - start}')

    np.save('know_face_encodings.npy', know_face_encodings)
    np.save('know_face_names.npy', know_face_names)

    print('\u001b[38;5;122mDone')


if __name__ == '__main__':
    save_faces()