import numpy as np
import os
import face_recognition as fr
from loguru import logger as log

@log.catch
def save_faces():
    """Збереження енкодингів облич із папки 'faces' у файли .npy"""
    know_face_encodings: list = []  # Список векторів облич
    know_face_names: list = []  # Список імен файлів (без розширення)

    for img in os.listdir('faces'):  # Перебір всіх зображень у папці 'faces'
        log.debug(f"Processing {img}")
        face_img = fr.load_image_file(f'faces/{img}')  # Завантаження зображення
        face_encoding = fr.face_encodings(face_img)[0]  # Отримання векторного представлення обличчя

        know_face_encodings.append(face_encoding)  # Додавання вектора до списку
        know_face_names.append(img[:-4])  # Видалення розширення файлу та збереження імені

    # Збереження отриманих даних у .npy файли
    np.save('data/know_face_encodings.npy', know_face_encodings)
    np.save('data/know_face_names.npy', know_face_names)

    print('\u001b[38;5;122mDone')  # Вивід повідомлення про завершення процесу

if __name__ == '__main__':
    save_faces()  # Виклик функції при запуску скрипта
