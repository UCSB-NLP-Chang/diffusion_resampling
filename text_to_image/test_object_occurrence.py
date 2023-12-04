import json
from PIL import Image
import argparse

import torch
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from utils import get_np_indices, load_spacy_stopwords


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--prompt_file", type=str, default='data/coco_object.json',
        help="File for text prmopts")
    parser.add_argument("--figure_dir", type=str, default='figures',
        help="Directory for figures")
    parser.add_argument("--device", type=str, default='cuda',
        help="Device to use")
    args = parser.parse_args()
    
    return args


def main():
    
    args = parse_args()
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101', pretrained=True).to(args.device)
    model.eval()

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(args.prompt_file, 'r') as f:
        texts = json.load(f)
        texts = [text.strip() for text in texts]
        texts = [text for text in texts if len(text) > 0]

    nlp, stopwords_list = load_spacy_stopwords()

    text_detect = []
    
    for i, text in tqdm(enumerate(texts)):
        img_path = f'{args.figure_dir}/{i}.png'
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(args.device)
        outputs = model(img)
        probs = outputs['pred_logits'].softmax(-1)
        
        _, _, coco_indices = get_np_indices(text, nlp, stopwords_list)
        coco_indices = torch.tensor(coco_indices, device=args.device)
        obj_probs = probs[0][:, coco_indices]
        obj_probs = obj_probs.max(0)[0]
        detect_i = (obj_probs > 0.5).sum().item()
        text_detect.append(detect_i / len(coco_indices))

    print(f"=============== Prompt file: {args.prompt_file} ===============")
    print(f"=============== Image directory: {args.figure_dir} ===============")
    print(f"Object occurrence: {np.mean(text_detect)}")


if __name__ == "__main__":
    main()
