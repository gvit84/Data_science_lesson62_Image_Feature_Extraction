import numpy as np
import cv2

def compute_edge_angles(image):
    # Застосовуємо фільтр Собеля для знаходження горизонтальних та вертикальних границь
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Обчислюємо амплітуду та напрямок границь
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_angle = np.arctan2(sobely, sobelx) * 180 / np.pi

    # Нормалізуємо значення амплітуди до діапазону [0, 255]
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return gradient_magnitude, gradient_angle

# Завантаження зображення
image = cv2.imread('dog1.jpg', cv2.IMREAD_GRAYSCALE)

# Визначення границь та кутів
edges, angles = compute_edge_angles(image)

# Відображення результатів
cv2.imshow('Edges', edges)
cv2.imshow('Angles', angles)
cv2.waitKey(0)
cv2.destroyAllWindows()
