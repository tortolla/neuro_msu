import numpy as np
import concurrent.futures
import logging
import os
import time
from PIL import Image
import re
from scipy.optimize import brentq  # Метод Брента для численного решения уравнений

# Файл для логирования
log_file = 'data_circles_new_400.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

# Коэффициенты для аппроксимаций
A_imp = -4.90752341e+00
B_imp = 1.61927247e+00
C_imp = -3.46762146e-03
D_imp = 2.71757824e-06

A_relax = 2.33142078e+02
B_relax = 2.96052741e-04
C_relax = 3.96503308e+01
D_relax = 2.96035007e-04

def impulse(conductivity, time_b):
    """Функция для расчета проводимости после импульса, с учетом положительного времени."""
    if conductivity is None:
        logging.error("Ошибка: conductivity равен None")
        raise ValueError("Conductivity is None")

    def func(dt):
        # Уравнение для нахождения корня по dt
        return A_imp + B_imp * dt + C_imp * dt**2 + D_imp * dt**3 - conductivity

    dt_min, dt_max = 0, 10000  # Ищем положительный корень, начиная с 0

    try:
        # Находим корень для положительного времени
        dt = brentq(func, dt_min, dt_max)
    except ValueError as e:
        logging.warning(f"Достигнуто максимальное значение для impulse: {e}")
        return 255  # Устанавливаем 255, если корень не найден в пределах диапазона

    # Добавляем time_b к найденному dt и пересчитываем проводимость
    total_time = dt + time_b
    up = A_imp + B_imp * total_time + C_imp * total_time**2 + D_imp * total_time**3
    return np.clip(up, 0, 255)

def relax(conductivity, time_b):
    """Функция для расчета проводимости при релаксации с учетом положительного времени."""
    if conductivity is None:
        logging.error("Ошибка: conductivity равен None")
        raise ValueError("Conductivity is None")

    def func(dt):
        # Уравнение для нахождения корня по dt
        return A_relax * np.exp(B_relax * (-dt)) + C_relax * np.exp(D_relax * (-dt)) - conductivity

    dt_min, dt_max = 0, 20000  # Ограничиваем диапазон до 20000

    try:
        # Находим корень для положительного времени
        dt = brentq(func, dt_min, dt_max)
    except ValueError as e:
        logging.warning(f"Достигнуто максимальное значение для relax: {e}")
        return 1e-10  # Устанавливаем минимальное значение, если корень не найден

    # Добавляем time_b к найденному dt и пересчитываем проводимость для релаксации
    total_time = dt + time_b
    down = A_relax * np.exp(B_relax * (-total_time)) + C_relax * np.exp(D_relax * (-total_time))
    return np.clip(down, 0, 255)

def get_sorted_image_paths(directory_path):
    """Функция для считывания и сортировки файлов из директории"""
    image_paths = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            image_paths.append(os.path.join(directory_path, filename))

    image_paths.sort(key=lambda x: int(re.search(r'^(\d+)', os.path.basename(x)).group(1)))

    return image_paths

def save_object_image(conductivity_array, folder_path, name):
    """Функция для сохранения изображения с текущими значениями Conductivity."""
    image = np.clip(conductivity_array, 0, 255).astype(np.uint8)  # Обрезаем значения и приводим к uint8
    img = Image.fromarray(image, 'L')

    if isinstance(name, int):
        name = str(name)

    if not os.path.splitext(name)[1]:
        name += '.png'

    img.save(os.path.join(folder_path, name))

def process_image_worker(args):
    """Вычисление реакции нейроморфного сенсора на участок изображения."""
    start_row, end_row, image_slice, conductivity_slice, time_b = args

    height, width = conductivity_slice.shape

    for i in range(height):
        for j in range(width):
            cond = conductivity_slice[i, j]
            
            if image_slice[i, j] != 0:
                new_cond = impulse(cond, time_b)
                conductivity_slice[i, j] = new_cond
            else:
                new_cond = relax(cond, time_b)
                conductivity_slice[i, j] = new_cond

    return start_row, end_row, conductivity_slice

if __name__ == '__main__':
    start_time = time.time()
    logging.info("Запуск программы")

    dir_path = input("Введите путь к директории с изображениями: ")
    time_b = int(input("Введите время между кадрами: "))
    n_processes = int(input(f"Введите количество используемых CPU (доступно {os.cpu_count()}): "))

    final_folder_name = os.path.basename(os.path.normpath(dir_path))
    folder_path_image = f"{final_folder_name}_time:{time_b}"
    
    log_file = f"data_{final_folder_name}_time{time_b}.log"

    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='w')

    logging.info("Запуск программы с параметрами:")
    logging.info(f"Путь к изображениям: {dir_path}")
    logging.info(f"Время между кадрами: {time_b}")
    logging.info(f"Количество используемых CPU: {n_processes}")

    path_array = get_sorted_image_paths(dir_path)
    sample_image = Image.open(path_array[0]).convert('L')
    width, height = sample_image.size
    logging.info(f"Размер изображения: {width}x{height}")

    os.makedirs(folder_path_image, exist_ok=True)
    logging.info(f"Путь для сохранения обработанных изображений: {folder_path_image}")

    conductivity_array = np.zeros((height, width), dtype=np.float32)

    print(f"Лог-файл: {log_file}")
    print(f"Путь для сохранения обработанных изображений: {folder_path_image}")

    rows_per_process = height // n_processes
    row_ranges = [(i * rows_per_process, (i + 1) * rows_per_process if i < n_processes - 1 else height)
                  for i in range(n_processes)]

    for idx, path in enumerate(path_array):
        print(f'Обработка изображения номер {idx}')
        logging.info(f"Обработка файла: {path}")
        name = os.path.basename(path)

        with Image.open(path).convert('L') as im:
            image_array = np.array(im, dtype=np.uint8)

        args_list = []
        for start_row, end_row in row_ranges:
            image_slice = image_array[start_row:end_row]
            conductivity_slice = conductivity_array[start_row:end_row].copy()
            args_list.append((start_row, end_row, image_slice, conductivity_slice, time_b))

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = [executor.submit(process_image_worker, args) for args in args_list]

            for future in concurrent.futures.as_completed(futures):
                start_row, end_row, updated_slice = future.result()
                conductivity_array[start_row:end_row] = updated_slice

        try:
            save_object_image(conductivity_array, folder_path_image, name)
            logging.info(f"Изображение {name} сохранено успешно.")
        except Exception as e:
            logging.error(f"Не удалось сохранить изображение {name}: {str(e)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Время выполнения программы: {elapsed_time} секунд")
    print(f"Время выполнения: {elapsed_time} секунд")

