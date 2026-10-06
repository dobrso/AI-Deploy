import os

import torch
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('HF_TOKEN')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MEMORY_CONVERSION = 1024 * 1024