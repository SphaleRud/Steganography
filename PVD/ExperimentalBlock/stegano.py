import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original, modified):
    score, diff = ssim(original, modified, full=True)
    return score


def get_capacity(diff, intervals):
    """
    Определяет вместимость блока (количество бит) для заданной разницы и интервалов.
    intervals: список кортежей (l, u, n)
    """
    for (l, u, n) in intervals:
        if l <= diff <= u:
            return (l, u, n)
    return None

def pvd_embed(image_array, secret_bits, intervals):
    """
    Встраивает секретные биты в изображение с использованием метода PVD.
    Возвращает изменённое изображение и количество встроенных бит.
    """
    rows, cols = image_array.shape
    bit_index = 0  # индекс текущего бита
    stego_image = image_array.copy()

    for i in range(rows):
        j = 0
        while j < cols - 1:
            p1 = int(stego_image[i, j])
            p2 = int(stego_image[i, j+1])
            d = abs(p1 - p2)

            cap = get_capacity(d, intervals)
            if cap is None:
                j += 2
                continue

            l, u, n = cap

            # Если оставшихся бит меньше, чем требуется, прекращаем встраивание
            if bit_index + n > len(secret_bits):
                return stego_image, bit_index

            bits_to_embed = secret_bits[bit_index:bit_index+n]
            b = int(bits_to_embed, 2)

            # Новое значение разницы d_new должно быть: d_new = l + b
            d_new = l + b

            # Корректировка значений пикселей для достижения разницы d_new
            if p1 >= p2:
                mid = (p1 + p2) // 2
                p1_new = mid + math.ceil(d_new / 2)
                p2_new = mid - math.floor(d_new / 2)
            else:
                mid = (p1 + p2) // 2
                p1_new = mid - math.floor(d_new / 2)
                p2_new = mid + math.ceil(d_new / 2)

            # Ограничение значений до [0, 255]
            p1_new = np.clip(p1_new, 0, 255)
            p2_new = np.clip(p2_new, 0, 255)

            stego_image[i, j] = p1_new
            stego_image[i, j+1] = p2_new

            bit_index += n
            j += 2

            if bit_index >= len(secret_bits):
                return stego_image, bit_index

    return stego_image, bit_index

def calculate_psnr(original, modified):
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return 100  # изображения идентичны
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def text_to_bits(text):
    """Преобразует текст в битовую строку."""
    bits = ''.join(bin(ord(c))[2:].zfill(8) for c in text)
    return bits

# Загрузка тестового изображения (grayscale)
image_path = 'grayscale.png'  # убедитесь, что файл существует
original_image = Image.open(image_path).convert('L')
image_array = np.array(original_image)

# Интервалы для PVD
intervals = [
    (0, 7, 3),
    (8, 15, 4),
    (16, 31, 5),
    (32, 63, 6),
    (64, 127, 7),
    (128, 255, 8)
]

# Определим максимальное количество бит, которое может быть встроено.
# Для этого пройдемся по всем парам без встраивания, суммируя вместимость.
max_capacity = 0
rows, cols = image_array.shape
for i in range(rows):
    j = 0
    while j < cols - 1:
        p1 = int(image_array[i, j])
        p2 = int(image_array[i, j+1])
        d = abs(p1 - p2)
        cap = get_capacity(d, intervals)
        if cap is not None:
            _, _, n = cap
            max_capacity += n
        j += 2

print(f"Максимальная вместимость изображения: {max_capacity} бит")

# Проведем эксперимент при разной загрузке: от 10% до 100% от max_capacity.
loadings = np.linspace(0.1, 1.0, 10)
psnr_values = []
embedded_bits_list = []

for load in loadings:
    bits_to_embed = int(max_capacity * load)
    # Генерируем случайную битовую строку требуемой длины
    secret_bits = ''.join(np.random.choice(['0','1']) for _ in range(bits_to_embed))
    
    stego_array, bits_used = pvd_embed(image_array, secret_bits, intervals)
    psnr = calculate_psnr(image_array, stego_array)
    psnr_values.append(psnr)
    embedded_bits_list.append(bits_used)
    print(f"Загрузка: {load*100:.0f}% — встроено бит: {bits_used}, PSNR: {psnr:.2f} dB")

    # Для наглядности можно сохранить несколько изображений при разной загрузке
    if load in [0.1, 0.5, 1.0]:
        Image.fromarray(np.uint8(stego_array)).save(f'stego_{int(load*100)}.png')

ssim_value = calculate_ssim(image_array, stego_array)
print(f"SSIM: {ssim_value:.4f}")

# Построение графиков
plt.figure(figsize=(12, 5))

# График PSNR от процента загрузки
plt.subplot(1, 2, 1)
plt.plot(loadings * 100, psnr_values, marker='o')
plt.xlabel('Загрузка (% от максимальной вместимости)')
plt.ylabel('PSNR (dB)')
plt.title('Зависимость PSNR от загрузки контейнера')
plt.grid(True)

# График количества встроенных бит от процента загрузки
plt.subplot(1, 2, 2)
plt.plot(loadings * 100, embedded_bits_list, marker='s', color='orange')
plt.xlabel('Загрузка (% от максимальной вместимости)')
plt.ylabel('Количество встроенных бит')
plt.title('Встроенная информация vs. загрузка контейнера')
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure()
plt.hist(np.abs(image_array.astype(int) - stego_array.astype(int)).ravel(), bins=50, color='blue', alpha=0.7)
plt.title("Распределение разностей пикселей")
plt.xlabel("Абсолютное изменение")
plt.ylabel("Частота")
plt.show()

diff_image = np.abs(image_array.astype(int) - stego_array.astype(int))
plt.imshow(diff_image, cmap='hot')
plt.title("Разностное изображение")
plt.colorbar()
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.hist(image_array.ravel(), bins=256, range=(0,255), color='gray')
plt.title("Гистограмма оригинала")
plt.subplot(1, 2, 2)
plt.hist(stego_array.ravel(), bins=256, range=(0,255), color='gray')
plt.title("Гистограмма стего-изображения")
plt.show()