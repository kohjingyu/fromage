#!/usr/bin/env python
# coding: utf-8

"""
This is a script reproducing the VisDial text-to-image (T2I) retrieval results from our paper,
Grounding Language Models to Images for Multimodal Inputs and Outputs (https://arxiv.org/abs/2301.13823).
This result is reported in Table 2 of the paper. This measures the recall of the model in selecting 
the appropriate image conditioned on a dialogue sequence.

Example usage: `python eval_visdial_retrieval.py`
"""

import numpy as np
import collections
import copy
import json
import os
import torch
from transformers import logging
from tqdm import notebook
logging.set_verbosity_error()

from PIL import Image
import matplotlib.pyplot as plt

from fromage import models
from fromage import utils


def get_pixel_values_from_path(path: str, feature_extractor):
    """Helper function for getting images pixels from a local path."""
    img = Image.open(path)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    pixel_values = utils.get_pixel_values_for_model(feature_extractor, img)
    if torch.cuda.is_available():
        pixel_values = pixel_values.bfloat16()
        pixel_values = pixel_values.cuda()
    return pixel_values[None, ...]


if __name__ == "__main__":
    # Load model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)


    # Download the VisDial validation annotations (https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0),
    # the dense answer annotations (https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0)
    # (for computing MRR) and the images (https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=0).
    # Extract everything to the `VisualDialog` folder.

    base_dir = 'VisualDialog/'
    split = 'val'
    img_dir = os.path.join(base_dir, f'VisualDialog_{split}2018')

    with open(os.path.join(base_dir, f'visdial_1.0_{split}.json'), 'r') as f:
        visdial_data = json.load(f)
        
    with open(os.path.join(base_dir, f'visdial_1.0_{split}_dense_annotations.json'), 'r') as f:
        dense_data = json.load(f)

    # Check that dense and sparse data are aligned.
    assert len(dense_data) == len(visdial_data['data']['dialogs'])
    for i in range(len(dense_data)):
        assert dense_data[i]['image_id'] == visdial_data['data']['dialogs'][i]['image_id']
        
    questions = visdial_data['data']['questions']
    answers = visdial_data['data']['answers']
    dialogs = visdial_data['data']['dialogs']

    # Then, we compute the image features and text features for each VisDial example:
    topk = (1, 5, 10)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none').cuda()

    all_visual_embs = []
    all_text_embs = []

    for example_idx in notebook.tqdm(range(len(dialogs))):
        dialog = dialogs[example_idx]
        image_id = str(dialog['image_id']).rjust(12, '0')
        contexts = []

        with torch.no_grad():
            images = get_pixel_values_from_path(
                os.path.join(img_dir, f'VisualDialog_{split}2018_{image_id}.jpg'),
                model.model.feature_extractor)
            visual_embs = model.model.get_visual_embs(images, mode='retrieval')

            for i in range(len(dialog['dialog'])):
                contexts.append('Q: ' + questions[dialog['dialog'][i]['question']] + '?')
                contexts.append('A: ' + answers[dialog['dialog'][i]['answer']] + '.')

            full_context_sent = ' '.join(contexts) + '[RET]'
            input_ids = model.model.tokenizer(full_context_sent, add_special_tokens=True, return_tensors="pt").input_ids
            input_ids = input_ids.cuda()
            input_embs = model.model.input_embeddings(input_ids)  # (N, T, D)
            generated_ids, output_embs, _ = model(input_embs, None, None, generate=True, num_words=1, temperature=0.0)
            embeddings = output_embs[0]

            full_input_ids = torch.cat([input_ids, generated_ids], dim=1)
            ret_emb = embeddings[:, -1, :]  

            all_visual_embs.append(visual_embs.cpu().detach().float().numpy())
            all_text_embs.append(ret_emb.cpu().detach().float().numpy())

    # Compute scores over the whole dataset:
    scores = np.concatenate(all_visual_embs, axis=0)[:, 0, :] @ np.concatenate(all_text_embs, axis=0).T
    scores = torch.tensor(scores).float()
    assert scores.shape == (2064, 2064), scores.shape


    # Finally, we can compute the Recall@k scores:
    _, preds = scores.topk(max(topk))
    for k in topk:
        labels = torch.arange(preds.shape[0])
        correct = torch.any(preds[:, :k] == labels[:, None], axis=1).sum()
        acc = correct / preds.shape[0]
        print(f'top-k, k={k}, acc={acc:.5f}')
    print('=' * 20)



