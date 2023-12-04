import re

from PIL import Image
import spacy
from nltk.corpus import stopwords
import inflect
from fuzzywuzzy import fuzz
import torch
from torch import nn
from diffusers import StableDiffusionPipeline

from schedulers import SDEDMScheduler
from t2i_discriminator import T2IDiscriminator


# COCO classes from DETR
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'ski',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def get_coco_names():
    engine = inflect.engine()
    classes = [[i.lower()] for i in COCO_CLASSES]
    # Add plural forms
    for each in classes:
        if each[0] == 'tv':
            each.append('television')
        if each[0] == 'person':
            each.append('people')
            each.append('man')
            each.append('woman')
            each.append('player')
            each.append('child')
        if each[0] != 'n/a':
            each.append(engine.plural(each[0]))
    
    return classes


def index_in_coco(noun, coco_names):
    noun = noun.lower()
    for idx, ni in enumerate(coco_names):
        if any([re.search(r'\b' + i + r'\b', noun) for i in ni]):
            return idx
        similarity = max([fuzz.token_set_ratio(noun, i) for i in ni])
        if similarity > 90:
            return idx
    return None


def get_np_indices(text_desc, nlp, stopwords, limit_coco=True):
    """Get noun phrase indices."""
    doc = nlp(text_desc)
    word_indices, noun_texts = [], []
    coco_indices = []
    if limit_coco:
        coco_names = get_coco_names()
    for chunk in doc.noun_chunks:
        noun = chunk.text
        if noun.lower() in stopwords:
            continue
        if limit_coco:
            coco_idx = index_in_coco(noun, coco_names)
            if coco_idx is not None:
                coco_indices.append(coco_idx)
                noun_texts.append((noun, coco_idx))
        word_indices.append((chunk.start, chunk.end - 1))
    
    return word_indices, noun_texts, list(set(coco_indices))


def load_diffusion_model(args):
    pipe = StableDiffusionPipeline.from_pretrained(args.diffusion_model_id, torch_dtype=torch.float32)
    assert pipe.scheduler.beta_schedule == 'scaled_linear'
    assert pipe.scheduler.prediction_type == 'epsilon'
    pipe.scheduler = SDEDMScheduler(pipe.unet, beta_min=pipe.scheduler.beta_start,
        beta_max=pipe.scheduler.beta_end, device=args.device)
    pipe.to(args.device)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.unet.eval()
    pipe.vae.eval()

    return pipe


def load_discriminator(args):
    if args.use_obj_discriminator:
        discriminator_ckpt = 'models/obj_discriminator.pt'
    elif args.method == 'pf-discriminator':
        discriminator_ckpt = 'models/discriminator_cond.pt'
    else:
        discriminator_ckpt = 'models/discriminator_uncond.pt'
    print(f"=============== Using discriminator {discriminator_ckpt} ===============")
    if args.use_obj_discriminator:
        discriminator = nn.Sequential(
            nn.Linear(92, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    else:
        discriminator = T2IDiscriminator.from_pretrained(
            args.diffusion_model_id, subfolder="unet", low_cpu_mem_usage=False, device_map=None)
    discriminator.load_state_dict(torch.load(discriminator_ckpt, map_location='cpu'))
    discriminator = discriminator.to(args.device)
    discriminator.eval()
    
    return discriminator


def load_spacy_stopwords():
    nlp = spacy.load("en_core_web_sm")
    stoplist = set(stopwords.words("english")).union(
        nlp.Defaults.stop_words
    )
    return nlp, stoplist
