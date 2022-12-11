import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, CenterCrop
import io
import clip
import webdataset as wds
from torch.utils.data import DataLoader
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = "cuda" if torch.cuda.is_available() else "cpu"

NSFW = ["porn", "NSFW", "sex", "sperm", "nipples", "breats", "tits", "boops", "penis", "dick", "cock", "clitoris", "vagina", "fuck", "fucking","lusty, horny", "lust, lusty", "horny", "sexual", "sexy", "sexy, sexual", "sexy, hot" , "hentai", "sexy drawing", "sexy painting", "sexy cartoon", "sexy pic"]
VIOLENCE = ["gun", "weapon","knife", "handcuff", "handcuffs", "tank, war", "pistol", "blood, bloody", "blade","rifle","horror", "gore", "terror"]

text_list_for_similarity_comparison = NSFW + VIOLENCE

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform():
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

custom_preproc_img = _transform()

def preprocopenclip224(img):
  image = Image.open(io.BytesIO(img)).convert('RGB')
  image = custom_preproc_img(image)
  return image

model, preprocess = clip.load("ViT-L/14", device=device)

text = clip.tokenize(text_list_for_similarity_comparison).to(device)
text_features = model.encode_text(text)

### Inference code
def inference_on_batch_col(b_col):
    with torch.no_grad():
        image_features = model.encode_image(b_col.to(device))
        
        logits_per_image, logits_per_text = model(b_col.to(device), text)
        text_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        ouput_image_features = image_features.cpu().numpy()

        return text_probs, ouput_image_features


### Test run
# inference_on_batch_col(b['jpg'])