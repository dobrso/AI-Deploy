import os
import time

import psutil
from datasets import load_dataset

from ai_deploy.models.image_model import init_image_model
from utils.settings import MEMORY_CONVERSION, TOKEN

process = psutil.Process(os.getpid())
memory_start = process.memory_info().rss / MEMORY_CONVERSION

model = init_image_model()

process.cpu_percent(interval=None)
memory_model = process.memory_info().rss / MEMORY_CONVERSION

image_dataset = 'ILSVRC/imagenet-1k'
split = 'validation'

N = 1000

dataset = load_dataset(image_dataset, split=split, token=TOKEN, streaming=True).take(N)
expected = []
predicted = []

top1_correct = 0
top5_correct = 0

label2id = model.model.config.label2id

t0 = time.time()

for item in dataset:
    image = item['image']
    expected.append(item['label'])

    image = image.convert('RGB')
    output = model(image)
    output_ids = [label2id[item['label']] for item in output]

    predicted.append(output_ids)

for i in range(len(expected)):
    if expected[i] == predicted[i][0]:
        top1_correct += 1
    if expected[i] in predicted[i]:
        top5_correct += 1

top1_accuracy = top1_correct / N
top5_accuracy = top5_correct / N
time_diff = time.time() - t0
memory_end = process.memory_info().rss / MEMORY_CONVERSION
cpu_usage = process.cpu_percent(interval=None)

print('=' * 60)
print(f'Количество примеров: {N}')
print(f'Точность top1: {top1_accuracy * 100:.02f}%')
print(f'Точность top5: {top5_accuracy * 100:.02f}%')
print(f'Время работы: {time_diff:.02f} сек.')
print(f'Среднее время на одно изображение: {time_diff / N:.02f} сек.')
print(f'Оперативная память на старте: {memory_start:.02f} MB')
print(f'Оперативная память после загрузки модели: {memory_model:.02f} MB')
print(f'Оперативная память в конце: {memory_end:.02f} MB')
print(f'Нагрузка на CPU: {cpu_usage:.02f}%')
print('=' * 60)