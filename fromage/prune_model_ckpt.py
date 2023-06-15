"""Prune pretrained model weights to reduce size.

This keeps only the weights that we finetune, and discards the pretrained LLM / visual encoder weights.
"""
import torch
import json

ckpt_path = 'ckpt.pth.tar'
pruned_output_path = 'ckpt_pruned.pth.tar'
model_args_path = 'model_args.json'

if __name__ == '__main__':
    with open(model_args_path, 'r') as f:
        model_kwargs = json.load(f)
        ret_token_idx = model_kwargs['retrieval_token_idx']

    with open(ckpt_path, 'rb') as f:
        checkpoint = torch.load(f)

    del checkpoint['epoch']
    del checkpoint['best_acc1']
    del checkpoint['optimizer']
    del checkpoint['scheduler']
    for k, v in checkpoint['state_dict'].items():
        checkpoint['state_dict'][k] = v.detach().clone()

    # Prune the pretrained token embeddings and keep just [RET].
    ret_embedding = checkpoint['state_dict']['module.model.input_embeddings.weight'][ret_token_idx:ret_token_idx+1, :].detach().clone()
    checkpoint['state_dict']['ret_input_embeddings.weight'] = ret_embedding

    with open(pruned_output_path, 'wb') as f:
        torch.save(checkpoint, f)