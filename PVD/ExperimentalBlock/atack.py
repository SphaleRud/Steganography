import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import skimage.util
import os


# Функция для расчёта PSNR
def calculate_psnr(original, modified):
    if original.shape != modified.shape:
        modified = resize_to_match(original, modified)
    return cv2.PSNR(original, modified)


# Функция для расчёта SSIM
def calculate_ssim(original, modified):
    if original.shape != modified.shape:
        modified = resize_to_match(original, modified)
    score, _ = ssim(original, modified, full=True)
    return score


# Функция загрузки изображения в grayscale
def load_image(image_path):
    return np.array(Image.open(image_path).convert("L"))


# Функция сохранения изображения
def save_image(image_array, filename):
    cv2.imwrite(filename, image_array)
    print(f"Сохранено: {filename}")


# Функция приведения размеров изображения к оригиналу
def resize_to_match(original, modified):
    return cv2.resize(modified, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)


# Функция построения гистограммы
def plot_histogram(original, modified, title, filename):
    plt.figure(figsize=(8, 4))
    plt.hist(original.ravel(), bins=256, alpha=0.5, label="Оригинал", color="blue")
    plt.hist(modified.ravel(), bins=256, alpha=0.5, label="Изменённое", color="red")
    plt.legend()
    plt.title(title)
    plt.xlabel("Интенсивность пикселей")
    plt.ylabel("Частота")
    plt.savefig(filename)
    plt.close()
    print(f"Сохранена гистограмма: {filename}")


# Создаём папку для сохранения результатов
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 📌 Загружаем изображения
original_image = load_image("original.png")
stego_image = load_image("stego_100.png")

# 📌 Применяем атаки и строим гистограммы
tests = [
    ("JPEG сжатие (качество 50)", cv2.imread(f"{output_dir}/compressed_50.jpg", cv2.IMREAD_GRAYSCALE), "jpeg_histogram.png"),
    ("Гауссовский шум (σ=15)", cv2.imread(f"{output_dir}/gaussian_noise.png", cv2.IMREAD_GRAYSCALE), "gaussian_histogram.png"),
    ("Salt & Pepper шум (0.05)", cv2.imread(f"{output_dir}/salt_pepper_noise.png", cv2.IMREAD_GRAYSCALE), "salt_pepper_histogram.png"),
    ("Размытие (Gaussian Blur)", cv2.imread(f"{output_dir}/blurred.png", cv2.IMREAD_GRAYSCALE), "blur_histogram.png"),
    ("Изменение размера (256x256)", cv2.imread(f"{output_dir}/resized_small.png", cv2.IMREAD_GRAYSCALE), "resize_small_histogram.png"),
    ("Изменение размера (1024x1024)", cv2.imread(f"{output_dir}/resized_large.png", cv2.IMREAD_GRAYSCALE), "resize_large_histogram.png"),
    ("Повышение контрастности", cv2.imread(f"{output_dir}/high_contrast.png", cv2.IMREAD_GRAYSCALE), "contrast_histogram.png")
]

print("\n📊 Результаты тестирования:")
for name, mod_img, hist_filename in tests:
    if mod_img is None:
        print(f"⚠ Ошибка: {name} не загружено! Проверьте, был ли сохранён файл.")
        continue

    psnr_value = calculate_psnr(stego_image, mod_img)
    ssim_value = calculate_ssim(stego_image, mod_img)
    print(f"{name} → PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")

    # Сохранение гистограммы
    plot_histogram(stego_image, mod_img, f"Гистограмма: {name}", f"{output_dir}/{hist_filename}")

print("\n✅ Тестирование завершено! Все результаты сохранены в папке 'output'.")
