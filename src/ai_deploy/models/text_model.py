from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

from utils.settings import TORCH_DTYPE, DEVICE

def init_text_model():
    model_name = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text_model = pipeline(
        'sentiment-analysis',
        model=model,
        tokenizer=tokenizer,
        dtype=TORCH_DTYPE,
        device=DEVICE
    )

    return text_model