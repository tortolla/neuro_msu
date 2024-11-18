import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ========== Параметры симуляции ==========

print("Input size of image:")
size = int(input())  # Размер игрового поля
print("Input radius:")
rad = int(input())  # Радиус шарика

boundary = size - rad

print("Input pixels per frame")
pixels_per_frame = int(input())  # Постоянная скорость

print("Input num frames")
num_frames = int(input())  # Количество кадров для симуляции

print("Output_folder:")
output_folder = input()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Параметры для контроля изменения направления
min_direction_change_steps = 2
max_direction_change_steps = 30

# ========== Класс для шариков ==========

class Ball:
    def __init__(self):
        # Начальные координаты и скорость
        self.position = np.array([random.uniform(rad, boundary), random.uniform(rad, boundary)], dtype=float)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = np.array([pixels_per_frame * math.cos(angle), pixels_per_frame * math.sin(angle)], dtype=float)
        self.direction_change_counter = random.randint(min_direction_change_steps, max_direction_change_steps)

    def move(self):
        # Плавное изменение направления через заданное количество шагов
        self.direction_change_counter -= 1
        if self.direction_change_counter <= 0:
            self.direction_change_counter = random.randint(min_direction_change_steps, max_direction_change_steps)
            angle_shift = random.uniform(-math.pi / 3, math.pi / 3)
            rotation_matrix = np.array([[math.cos(angle_shift), -math.sin(angle_shift)],
                                        [math.sin(angle_shift), math.cos(angle_shift)]])
            self.velocity = rotation_matrix @ self.velocity
            self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * pixels_per_frame

        # Обновляем позицию
        self.position += self.velocity
        self.check_boundary()

    def check_boundary(self):
        if self.position[0] <= rad or self.position[0] >= boundary:
            self.velocity[0] = -self.velocity[0]
        if self.position[1] <= rad or self.position[1] >= boundary:
            self.velocity[1] = -self.velocity[1]

def check_collisions(balls):
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            ball1, ball2 = balls[i], balls[j]
            distance = np.linalg.norm(ball1.position - ball2.position)
            if distance <= 2 * rad:
                # Обработка столкновения: изменение направления скоростей
                direction = (ball1.position - ball2.position) / distance
                velocity1_parallel = np.dot(ball1.velocity, direction) * direction
                velocity2_parallel = np.dot(ball2.velocity, direction) * direction
                ball1.velocity -= 2 * velocity1_parallel
                ball2.velocity -= 2 * velocity2_parallel

# Создаем шарики
balls = [Ball() for _ in range(3)]

# Настраиваем фигуру для сохранения изображений
fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')


# Генерация и сохранение кадров
for frame_num in tqdm(range(1, num_frames + 1), desc="Сохранение изображений"):
    ax.clear()
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    plt.axis('off')

    # Проверяем и обрабатываем столкновения
    check_collisions(balls)

    # Двигаем шарики и добавляем их на рисунок
    for ball in balls:
        ball.move()
        circle = plt.Circle(ball.position, rad, color='white')
        ax.add_artist(circle)

    # Формируем имя файла и сохраняем изображение
    coords = []
    for ball in balls:
        x, y = int(ball.position[0]), int(ball.position[1])
        coords.extend([x, y])
    coords_str = '_'.join(map(str, coords))
    filename = f"{frame_num}_{coords_str}.png"
    plt.savefig(os.path.join(output_folder, filename), pad_inches=0)
