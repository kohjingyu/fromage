# This is a script reproducing the contextual image retrieval results from our paper.
# This result is reported in Table 1.
# The results of this may be slightly different from the paper, as the Flickr images from Visual Storytelling may disappear over time.

import numpy as np
import collections
import copy
import json
import os
import torch
from transformers import logging
from tqdm import tqdm
logging.set_verbosity_error()

from PIL import Image
import matplotlib.pyplot as plt

from fromage import models
from fromage import utils


# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

# Download the Visual Storytelling SIS dataset from https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz
# Extract the files (there should be three sets: train, val, and test).
# We use the val set for reporting results.
vist_val_json_path = 'sis/val.story-in-sequence.json'
with open(vist_val_json_path, 'r') as f:
    vist_data_raw = json.load(f)
    
# Format into a dictionary of {story_id: data} items.
vist_data = {
    'annotations': collections.defaultdict(list)
}
used_image_ids = []


for ann in vist_data_raw['annotations']:
    assert len(ann) == 1
    ann = ann[0]
    story_id = ann['story_id']
    vist_data['annotations'][story_id].append({
        'caption': ann['text'],
        'image_id': ann['photo_flickr_id'],
        'sequence_index': ann['worker_arranged_photo_order'],
    })
    used_image_ids.append(ann['photo_flickr_id'])

used_image_ids = set(used_image_ids)
print(len(used_image_ids))


# Precompute image features for running retrieval.
embs_fn = 'sis_img_features.npy'
id2url = {}

for image_data in vist_data_raw['images']:
    image_id = image_data['id']
    if image_id in used_image_ids:
        image_url = image_data.get('url_o', None)
        if image_url is not None:
            id2url[image_id] = image_url

if not os.path.exists(embs_fn):
    print(f'{embs_fn} does not exist, computing it from scratch.')
    all_visual_embs = []
    all_image_ids = []

    for image_id, image_url in tqdm(id2url.items()):
        try:
            images = utils.get_image_from_url(image_url)
            pixel_values = utils.get_pixel_values_for_model(model.model.feature_extractor, images)
            pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
            pixel_values = pixel_values[None, ...]
            visual_embs = model.model.get_visual_embs(pixel_values, mode='retrieval')
            all_visual_embs.append(visual_embs.float().cpu().detach().numpy())
            all_image_ids.append(image_id)
        except Image.UnidentifiedImageError:
            pass

    all_image_ids = np.array(all_image_ids)
    all_visual_embs = np.concatenate(all_visual_embs, axis=0)
    assert all_image_ids.shape[0] == all_visual_embs.shape[0], (all_image_ids.shape, all_visual_embs.shape)
    print(all_image_ids.shape, all_visual_embs.shape)

    with open(embs_fn, 'wb') as wf:
        np.save(wf, {'image_ids': all_image_ids, 'embeddings': all_visual_embs})

# Load embeddings.
with open(embs_fn, 'rb') as wf:
    embs_data = np.load(wf, allow_pickle=True).item()
    all_image_ids = embs_data['image_ids']
    emb_matrix = embs_data['embeddings']

# Normalize embedding matrix to be suitable for image retrieval.
logit_scale = model.model.logit_scale.exp()
emb_matrix = torch.tensor(emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
emb_matrix = emb_matrix / emb_matrix.norm(dim=-1, keepdim=True)
emb_matrix = logit_scale * emb_matrix
print('emb_matrix.shape', emb_matrix.shape)


# Then, for each VIST example, we process it as `<caption1><img1><caption2><img2>...<caption5> [RET]`,
# providing this as input to FROMAGe, and retrieve the image corresponding to the `[RET]` embedding.

topk = (1, 5, 10)
top_k_preds = {}

with torch.no_grad():
    for story_idx, (story_id, story_data) in tqdm(enumerate(vist_data['annotations'].items()), total=len(vist_data['annotations'])):
        gt_image_id = story_data[-1]['image_id']
        skip = False  # Skip examples that do not have images (due to URLs being taken down, or something)
        for s in story_data:
            if s['image_id'] not in all_image_ids or s['image_id'] not in id2url:
                skip = True
                break

        if not skip:
            # Use the first n-1 images and n captions as input.
            image_urls = [id2url[s['image_id']] for s in story_data[:-1]]
            captions = [s['caption'] for s in story_data]
            assert len(image_urls) == len(captions) - 1

            visual_embs = []
            # Compute embeddings for the input images.
            images = [utils.get_image_from_url(image_url) for image_url in image_urls]
            pixel_values = [utils.get_pixel_values_for_model(model.model.feature_extractor, image) for image in images]
            pixel_values = torch.stack(pixel_values, dim=0)  # (n-1, 3, 224, 224)
            pixel_values = pixel_values.to(device=model.model.logit_scale.device, dtype=model.model.logit_scale.dtype)
            visual_embs = model.model.get_visual_embs(pixel_values, mode='captioning')

            # Compute embeddings for the input captions.
            all_input_ids = []
            for i, c in enumerate(captions):
                if i == len(captions) - 1:
                    c += '[RET]'  # Add the [RET] token to the final caption.
                input_ids = model.model.tokenizer(c, add_special_tokens=True, return_tensors="pt").input_ids.to(emb_matrix.device)
                all_input_ids.append(input_ids)
            
            input_embs = [model.model.input_embeddings(s)[0, ...] for s in all_input_ids]  # (N, T, D)

            # Interleave captions and images as [caption1, image1, caption2, ..., image4, caption5].
            final_input_embs = []
            assert len(visual_embs) == len(input_embs) - 1
            for i in range(len(images)):
                final_input_embs.append(input_embs[i])
                final_input_embs.append(visual_embs[i])
            final_input_embs.append(input_embs[len(images)])
            final_input_embs = torch.cat(final_input_embs, dim=0)[None, ...]  # (1, T, 4096)
            
            # Get embedding of the [RET] token, and compute scores:
            output = model.model.lm(inputs_embeds=final_input_embs, labels=None, use_cache=False, output_hidden_states=True)
            last_hidden_state = model.model.text_hidden_fcs[0](output.hidden_states[-1])
            ret_emb = last_hidden_state[:, -1, :]

            ret_emb = ret_emb / ret_emb.norm(dim=1, keepdim=True)
            scores = ret_emb.squeeze() @ emb_matrix.squeeze().T
            
            # Don't retrieve previously seen images.
            prev_image_ids = [s['image_id'] for s in story_data[:-1]]
            for prev_id in prev_image_ids:
                scores[np.where(all_image_ids == prev_id)[0]] -= 10000
            
            # Store top-k preds.
            _, preds = scores.topk(max(topk))
            preds = preds.cpu().detach().numpy()
            preds = [all_image_ids[p] for p in preds]
            top_k_preds[story_id] = {'topk_preds': preds, 'gt': gt_image_id}


# Finally, we can compute Recall@k:
top_k_accuracy = collections.defaultdict(list)

for story_id, results in top_k_preds.items():
    for k in topk:
        acc = results['gt'] in results['topk_preds'][:k]
        top_k_accuracy[k].append(acc)

for k in topk:
    result_str = f'k={k}, acc={np.mean(top_k_accuracy[k]):.5f}'
    print(result_str)
