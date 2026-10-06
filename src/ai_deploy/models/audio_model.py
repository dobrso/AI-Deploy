from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils.settings import DEVICE, TORCH_DTYPE

def init_audio_model():
    model_name = 'openai/whisper-large-v3-turbo'

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_name)

    audio_model = pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    return audio_model