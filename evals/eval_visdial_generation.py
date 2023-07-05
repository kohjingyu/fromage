#!/usr/bin/env python
# coding: utf-8

"""
This is a script reproducing the VisDial image-and-text-to-image (IT2I) results from our paper,
Grounding Language Models to Images for Multimodal Inputs and Outputs (https://arxiv.org/abs/2301.13823).
This result is reported in Table 2 of the paper.  This is the standard VisDial (https://arxiv.org/abs/1611.08669)
evaluation, which measures the ability of models to pick out the correct text answer out of 100 options.
This script reports NDCG, MRR, and R@k results.

Example usage: `python eval_visdial_generation.py`
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


# Parameters used for eval.
topk = (1, 5, 10)
# Number of options in a batch to compute loss for.
# If using a GPU with lower VRAM, this may have to be lowered.
batch_size = 20

# Download the VisDial validation annotations (https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0),
# the dense answer annotations (https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0)
# (for computing MRR) and the images (https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=0).
# Extract everything to the `VisualDialog` folder.
# First, we'll load the annotations, and define the paths to our images
# and annotations:
base_dir = 'VisualDialog/'
split = 'val'

# Path to save intermediate results to, to allow resuming in case of interruptions.
save_path = 'visdial_results_full.npy'



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


# Define some classes to help us compute NDCG and MRR.
# Modified from https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch/blob/master/visdialch/metrics.py
class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def observe(
            self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.
        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze(1)
        batch_size, num_options = predicted_ranks.size()

        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rnk.
        _, best_rankings = torch.sort(
            target_relevance, dim=-1, descending=True
        )

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True, key=""):
        if self._ndcg_denominator > 0:
            metrics = {
                key + "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0
        

def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this
    # position but we want i-th position to have rank of score at that
    # position, do this conversion
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks


if __name__ == "__main__":
    # Load model used in the paper.
    model_dir = './fromage_model/'
    model = models.load_fromage(model_dir)

    # Load VisDial data.
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

    # Then, for each VisDial example, we compute the loss 
    # conditioned on the image and the preceding dialogue. 
    # We return the option with the lowest loss as the answer:
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none').cuda()

    if os.path.exists(save_path):
        with open(save_path, 'rb') as rf:
            all_data = np.load(rf, allow_pickle=True).item()
            all_preds = all_data['all_preds']
            all_gt_results = all_data['all_gt_results']
            all_losses = all_data['all_losses']
            assert len(all_preds) == len(all_gt_results) == len(all_losses)
    else:
        # No in progress data, initialize from scratch.
        all_preds = []
        all_gt_results = []
        all_losses = []

    for example_idx in notebook.tqdm(range(len(all_preds) // 10, len(dialogs))):
        dialog = dialogs[example_idx]
        image_id = str(dialog['image_id']).rjust(12, '0')
        contexts = []

        with torch.no_grad():
            images = get_pixel_values_from_path(
                os.path.join(img_dir, f'VisualDialog_{split}2018_{image_id}.jpg'),
                model.model.feature_extractor)
            visual_embs = model.model.get_visual_embs(images, mode='captioning')

            for i in range(len(dialog['dialog'])):
                prev_d = dialog['dialog'][i-1]
                current_d = dialog['dialog'][i]
                if i > 0:
                    contexts.append('A: ' + answers[prev_d['answer']])
                contexts.append('Q: ' + questions[current_d['question']] + '?')
                answer_options = [answers[i] for i in current_d['answer_options']]
                answer = answers[current_d['answer']]
                gt_index = current_d['gt_index']
                caption = '\n'.join(contexts) + '\nA: '

                # Run through every possible option, and pick the option with 
                # the lowest loss (= lowest perplexity)
                example_losses = []
                # Tokenize the dialogue sequence (as this is the same for all answer choices).
                caption_ids = model.model.tokenizer(
                    caption, add_special_tokens=True, return_tensors="pt").input_ids
                caption_ids = caption_ids.to(images.device)
                caption_embs = model.model.input_embeddings(caption_ids)  # (N, T, D)
                condition_length = visual_embs.shape[1] + caption_embs.shape[1]

                all_example_embs = []
                all_example_labels = []

                for _, ans in enumerate(answer_options):
                    ans_ids = model.model.tokenizer(ans, add_special_tokens=True, return_tensors="pt").input_ids
                    ans_ids = ans_ids.to(images.device)
                    ans_embs = model.model.input_embeddings(ans_ids)
                    input_embs = torch.cat([
                        visual_embs,
                        caption_embs,
                        ans_embs], dim=1)
                    labels = torch.cat([
                        torch.zeros(visual_embs.shape[:-1], device=caption_ids.device, dtype=caption_ids.dtype) - 100,
                        caption_ids,
                        ans_ids], dim=1)
                    assert labels.shape[1] == input_embs.shape[1]

                    all_example_embs.append(input_embs)
                    all_example_labels.append(labels)

                max_len = max([x.shape[1] for x in all_example_labels])
                padded_example_embs = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[1])) for x in all_example_embs]
                padded_example_embs = torch.cat(padded_example_embs, axis=0)

                padded_example_labels = [torch.nn.functional.pad(x, (0, max_len - x.shape[1]), value=-100) for x in all_example_labels]
                padded_example_labels = torch.cat(padded_example_labels, axis=0)

                all_logits = []
                batches = int(np.ceil(padded_example_embs.shape[0] / batch_size))
                for i in range(batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    out = model.model.lm(
                        inputs_embeds=padded_example_embs[start_idx:end_idx, ...],
                        labels=None,
                        use_cache=False,
                        output_hidden_states=True)
                    all_logits.append(out.logits)

                logits = torch.cat(all_logits, dim=0)
                example_losses = ce_loss(logits.reshape((-1, logits.shape[-1])), padded_example_labels.reshape((-1,)))
                example_losses = example_losses.reshape((100, max_len))[:, condition_length:]
                example_losses = example_losses.sum(axis=1)

                all_losses.append(example_losses.cpu().float().numpy())
                scores = -example_losses
                _, preds = scores.topk(max(topk))
                all_preds.append(preds)
                all_gt_results.append(gt_index)

        with open(save_path, 'wb') as wf:
            np.save(wf, {'all_preds': all_preds, 'all_gt_results': all_gt_results, 'all_losses': all_losses})

    # Finally, we can compute NDCG, MRR, and Recall@k:
    with open(save_path, 'rb') as rf:
        all_data = np.load(rf, allow_pickle=True).item()
        all_preds = all_data['all_preds']
        all_gt_results = all_data['all_gt_results']
        all_losses = all_data['all_losses']

    top_k_accuracy = collections.defaultdict(list)
    mrr_results = []
    all_ranks = []
    topk = (1, 5, 10, 20)
    ndcg = NDCG()

    assert len(all_preds) == len(all_gt_results)
    for gt, loss in zip(all_gt_results, all_losses):
        scores = -loss
        _, preds = torch.tensor(scores).topk(100)
        rank = np.where(preds == gt)[0][0] + 1
        all_ranks.append(rank)
        mrr_results.append(1 / rank)

        for k in topk:
            acc = gt in preds[:k]
            top_k_accuracy[k].append(acc)
            
    dense_mrr = []
    for i in range(len(dense_data)):
        idx = i * 10 + dense_data[i]['round_id']
        if idx >= len(all_losses):
            break
        scores = -torch.tensor(all_losses[idx])[None, :]
        relevance = torch.tensor(dense_data[i]['gt_relevance'])[None, :]
        ndcg.observe(scores, relevance)
        dense_mrr.append(mrr_results[idx])

    for k in topk:
        print(f'top-k, k={k}, acc={np.mean(top_k_accuracy[k]):.5f}')
    print(f'MRR: {np.mean(mrr_results):.5f}')
    print(f'NDCG: {ndcg.retrieve(reset=True)["ndcg"]:.5f}')
