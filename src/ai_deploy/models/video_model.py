import torch

from utils.settings import DEVICE

def init_video_model():
    model_name = 'slowfast_r50'

    video_model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    video_model = video_model.eval().to(DEVICE)

    return video_model