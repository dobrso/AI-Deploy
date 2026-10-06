import os
import time

import psutil
from datasets import load_dataset
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from sklearn.metrics import accuracy_score

from ai_deploy.models.audio_model import init_audio_model
from utils.settings import MEMORY_CONVERSION

process = psutil.Process(os.getpid())
memory_start = process.memory_info().rss / MEMORY_CONVERSION

model = init_audio_model()

process.cpu_percent(interval=None)
memory_model = process.memory_info().rss / MEMORY_CONVERSION

audio_dataset = 'fixie-ai/common_voice_17_0'
language = 'ru'
split = 'test'

N = 50

dataset = load_dataset(audio_dataset, language, split=split)
expected = []
predicted = []

t0 = time.time()

for i in range(min(N, len(dataset))):
    output = model(
        dataset[i]['audio'],
        generate_kwargs={'task': 'transcribe'}
    )
    predicted.append(output['text'].strip())
    expected.append(dataset[i]['sentence'].strip())

norm = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

predicted_norm = [norm(x) for x in predicted]
expected_norm = [norm(x) for x in expected]

accuracy = accuracy_score(expected_norm, predicted_norm)
wer_metric = wer(expected_norm, predicted_norm)
cer_metric = cer(expected_norm, predicted_norm)
time_diff = time.time() - t0
memory_end = process.memory_info().rss / MEMORY_CONVERSION
cpu_usage = process.cpu_percent(interval=None)

print('=' * 60)
print(f'Количество примеров: {N}')
print(f'Точность: {accuracy * 100:.02f}%')
print(f'WER: {wer_metric * 100:.02f}%')
print(f'CER: {cer_metric * 100:.02f}%')
print(f'Время работы: {time_diff:.02f} сек.')
print(f'Средняя время на одно аудио: {time_diff / N:.02f} сек.')
print(f'Оперативная память на старте: {memory_start:.02f} MB')
print(f'Оперативная память после загрузки модели: {memory_model:.02f} MB')
print(f'Оперативная память в конце: {memory_end:.02f} MB')
print(f'Нагрузка на CPU: {cpu_usage:.02f}%')
print('=' * 60)