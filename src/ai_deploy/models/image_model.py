from transformers import ResNetForImageClassification, AutoImageProcessor, pipeline

from utils.settings import DEVICE, TORCH_DTYPE

def init_image_model():
    model_name = 'microsoft/resnet-34'

    model = ResNetForImageClassification.from_pretrained(
        model_name,
        dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(DEVICE)

    processor = AutoImageProcessor.from_pretrained(model_name)

    image_model = pipeline(
        'image-classification',
        model=model,
        image_processor=processor,
        dtype=TORCH_DTYPE,
        device=DEVICE
    )

    return image_model