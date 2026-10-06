import os
import time

import psutil
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

from ai_deploy.models.text_model import init_text_model
from utils.settings import MEMORY_CONVERSION

process = psutil.Process(os.getpid())
memory_start = process.memory_info().rss / MEMORY_CONVERSION

model = init_text_model()

process.cpu_percent(interval=None)
memory_model = process.memory_info().rss / MEMORY_CONVERSION

dataset_name = 'cardiffnlp/tweet_sentiment_multilingual'
language = 'english'
split = 'train'

N = 1500

dataset = load_dataset(dataset_name, language, split=split)
expected = []
predicted = []

label2id = model.model.config.label2id

t0 = time.time()

for i in range(min(N, len(dataset))):
    output = model(dataset[i]['text'], max_length=512)
    predicted.append(label2id[output[0]['label']])
    expected.append(dataset[i]['label'])

accuracy = accuracy_score(expected, predicted)
time_diff = time.time() - t0
memory_end = process.memory_info().rss / MEMORY_CONVERSION
cpu_usage = process.cpu_percent(interval=None)

print('=' * 60)
print(f'Количество примеров: {N}')
print(f'Точность: {accuracy * 100:.02f}%')
print(f'Время работы: {time_diff:.02f} сек.')
print(f'Среднее время на один текст: {time_diff / N:.02f} сек.')
print(f'Оперативная память на старте: {memory_start:.02f} MB')
print(f'Оперативная память после загрузки модели: {memory_model:.02f} MB')
print(f'Оперативная память в конце: {memory_end:.02f} MB')
print(f'Нагрузка на CPU: {cpu_usage:.02f}%')
print('=' * 60)
print(classification_report(
    expected, predicted,
    target_names=['negative', 'neutral', 'positive'],
    digits=4,
))
print('=' * 60)