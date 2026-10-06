import os
import json
import time
import tempfile

import psutil
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey,ShortSideScale,UniformTemporalSubsample
from pytorchvideo.data.encoded_video import EncodedVideo
from datasets import load_dataset, Video

from utils.settings import MEMORY_CONVERSION, DEVICE
from ai_deploy.models.video_model import init_video_model

process = psutil.Process(os.getpid())
memory_start = process.memory_info().rss / MEMORY_CONVERSION

model = init_video_model()

process.cpu_percent(interval=None)
memory_model = process.memory_info().rss / MEMORY_CONVERSION

N = 50
CLIP_DURATION = 2.0

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
alpha = 4

class PackPathway(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        return [slow_pathway, fast_pathway]


transform = ApplyTransformToKey(
    key='video',
    transform=Compose([
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean, std),
        ShortSideScale(size=side_size),
        CenterCropVideo(crop_size=(crop_size, crop_size)),
        PackPathway(),
    ]),
)

dataset_name = 'nateraw/kinetics-mini'
split = 'validation'

dataset = load_dataset(dataset_name, split=split)
dataset = dataset.cast_column('video', Video(decode=False))

id2label = {0: 'archery', 1: 'bowling', 2: 'flying_kite', 3: 'high_jump', 4: 'marching'}

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
local_path = os.path.join(project_root, 'kinetics_classnames.json')

with open(local_path) as f:
    kinetics_classnames = json.load(f)

model_id2label = {
    v: str(k).replace('"', '')
    for k, v in kinetics_classnames.items()
}

def predict_video(video_path, model, transform, device, top_k=5):
    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(start_sec=0, end_sec=CLIP_DURATION)
    video_data = transform(video_data)

    inputs = video_data['video']
    inputs = [i.to(device)[None, ...] for i in inputs]

    with torch.no_grad():
        preds = model(inputs)
        preds = F.softmax(preds, dim=1)

    top_probs, top_ids = torch.topk(preds, top_k)
    return top_ids[0].cpu().tolist(), top_probs[0].cpu().tolist()

t0 = time.time()

predicted = []
expected = []

for i, item in enumerate(dataset):
    if i >= N:
        break

    video_field = item['video']
    if isinstance(video_field, dict) and video_field.get('path'):
        video_path = video_field['path']
    else:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_field['bytes'])
            video_path = tmp.name

    top5_ids, top5_probs = predict_video(video_path, model, transform, DEVICE)

    top5_labels = [model_id2label[id] for id in top5_ids]

    predicted.append(top5_labels)
    expected.append(id2label[item['label']])

top1_correct = 0
top5_correct = 0

for i in range(N):
    if predicted[i][0] == expected[i]:
        top1_correct += 1
    if expected[i] in predicted[i]:
        top5_correct += 1

top1_accuracy = top1_correct / N
top5_accuracy = top5_correct / N
time_diff = time.time() - t0
memory_end = process.memory_info().rss / MEMORY_CONVERSION
cpu_usage = process.cpu_percent(interval=None)

print('=' * 60)
print(f'Количество элементов: {N}')
print(f'Точность top1: {top1_accuracy * 100:.02f}%')
print(f'Точность top5: {top5_accuracy * 100:.02f}%')
print(f'Время работы: {time_diff:.02f} сек.')
print(f'Среднее время на один видеофрагмент: {time_diff / N:.02f} сек.')
print(f'Оперативная память в начале: {memory_start:.02f} MB')
print(f'Оперативная память после загрузки модели: {memory_model:.02f} MB')
print(f'Оперативная память в конце: {memory_end:.02f} MB')
print(f'Нагрузка на CPU: {cpu_usage:.02f}%')
print('=' * 60)