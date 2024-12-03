import numpy as np
from PIL import Image

def embed_message(img, message):
    # Преобразуем сообщение в двоичный вид
    binary_message = ''.join([format(ord(char), '08b') for char in message]) + '1111111111111110'  # Стоп-сигнал
    message_index = 0
    img_array = np.array(img)
    
    for i in range(0, img_array.shape[0], 2):  # Пары пикселей
        for j in range(0, img_array.shape[1], 2):
            if message_index >= len(binary_message):  # Прекращаем, если все биты сообщения вставлены
                break

            p1 = img_array[i, j]
            p2 = img_array[i, j+1] if j + 1 < img_array.shape[1] else p1

            # Ограничиваем разницу, чтобы избежать переполнения
            diff = abs(p1 - p2)
            diff = min(diff, 255)

            bits_to_hide = 1 if diff < 10 else 2 if diff < 30 else 3
            hidden_bits = binary_message[message_index:message_index + bits_to_hide]
            message_index += bits_to_hide

            new_diff = int(hidden_bits, 2)

            # Обновляем пиксели, чтобы они оставались в пределах диапазона [0, 255]
            if p1 > p2:
                p1 = max(0, min(255, p1 - (diff - new_diff)))  # Ограничиваем диапазон от 0 до 255
                p2 = max(0, min(255, p2))
            else:
                p1 = max(0, min(255, p1))
                p2 = max(0, min(255, p2 - (diff - new_diff)))  # Ограничиваем диапазон от 0 до 255

            # Применяем изменения в изображение
            img_array[i, j], img_array[i, j+1] = p1, p2

    return Image.fromarray(img_array)

def extract_message(img):
    img_array = np.array(img)
    binary_message = ""
    
    for i in range(0, img_array.shape[0], 2):
        for j in range(0, img_array.shape[1], 2):
            p1 = img_array[i, j]
            p2 = img_array[i, j+1] if j + 1 < img_array.shape[1] else p1

            # Ограничиваем разницу, чтобы избежать переполнения
            diff = abs(p1 - p2)
            diff = min(diff, 255)

            # Определяем количество бит, которые можно извлечь из разницы
            bits_to_extract = 1 if diff < 10 else 2 if diff < 30 else 3

            # Извлекаем биты разницы
            extracted_bits = format(diff, '0' + str(bits_to_extract) + 'b')
            binary_message += extracted_bits

            # Проверяем, достигли ли мы стоп-сигнала
            if binary_message[-16:] == '1111111111111110':  # Стоп-символ
                break

    # Преобразуем двоичное сообщение в текст
    message = ''.join(chr(int(binary_message[i:i + 8], 2)) for i in range(0, len(binary_message) - 16, 8))
    return message


img = Image.open('original_image_black.png').convert('L')  # Загружаем изображение в серых тонах
message = "Hello, World!"  # Сообщение для встраивания
stego_image = embed_message(img, message)  # Встраиваем сообщение
stego_image.save('stego_image.png')  # Сохраняем изображение с сообщением

extracted_message = extract_message(stego_image)  # Извлекаем сообщение
print("Извлеченное сообщение:", extracted_message)


