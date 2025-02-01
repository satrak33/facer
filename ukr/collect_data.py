import numpy as np
import os
import face_recognition as fr

# Функція для збереження кодованих даних обличчя з папки 'faces'
def save_faces():
    # Списки для зберігання кодованих даних облич та їх імен
    know_face_encodings: list = []
    know_face_names: list = []

    # Перебір усіх файлів у папці 'faces'
    for img in os.listdir('../faces'):

        # Завантаження зображення обличчя
        face_img = fr.load_image_file(f'faces/{img}')
        # Отримання кодованих даних першого обличчя на зображенні
        face_encoding = fr.face_encodings(face_img)[0]

        # Додавання даних до відповідних списків
        know_face_encodings.append(face_encoding)
        know_face_names.append(img)

    # Збереження списків у файли .npy для подальшого використання
    np.save('know_face_encodings.npy', know_face_encodings)
    np.save('know_face_names.npy', know_face_names)

    # Повідомлення про успішне виконання
    print('\u001b[38;5;122mВиконано')

# Головна функція програми
if __name__ == '__main__':
    save_faces()
