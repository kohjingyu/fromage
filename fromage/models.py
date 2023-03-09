from typing import Callable, List, Optional, Tuple, Union
from collections import namedtuple
import json
import glob
import math
import numpy as np
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import pickle as pkl
from PIL import Image, UnidentifiedImageError

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import OPTForCausalLM, GPT2Tokenizer
from transformers import CLIPVisionModel, CLIPVisionConfig

from fromage import utils


class FrozenArgs:
  freeze_lm: bool = True
  freeze_vm: bool = True
  opt_version: str = 'facebook/opt-6.7b'
  visual_encoder: str = 'openai/clip-vit-large-patch14'
  n_visual_tokens: int = 1
  image_embed_dropout_prob: float = 0.0
  task: str = 'captioning'
  shared_emb_dim: Optional[int] = 256
  text_emb_layers: List[int] = [-1]
  retrieval_token_idx: int = 0


class FromageModel(nn.Module):
  def __init__(self, tokenizer, args: FrozenArgs = FrozenArgs()):
    super().__init__()
    self.tokenizer = tokenizer
    self.feature_extractor = utils.get_feature_extractor_for_model(args.visual_encoder, train=False)
    self.image_token = self.tokenizer.cls_token_id
    assert args.text_emb_layers != set(args.text_emb_layers), 'text_emb_layers not unique'
    self.args = args

    opt_version = args.opt_version
    visual_encoder = args.visual_encoder
    n_visual_tokens = args.n_visual_tokens
    print(f"Using {opt_version} for the language model.")
    print(f"Using {visual_encoder} for the visual model with {n_visual_tokens} visual tokens.")

    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    if 'facebook/opt' in opt_version:
      self.lm = OPTForCausalLM.from_pretrained(opt_version)
    else:
      raise NotImplementedError

    self.opt_version = opt_version

    if self.args.freeze_lm:
      self.lm.eval()
      print("Freezing the LM.")
      for param in self.lm.parameters():
        param.requires_grad = False
    else:
      self.lm.train()

    self.retrieval_token_idx = args.retrieval_token_idx
    print(f'Initializing embedding for the retrieval token [RET] (id = {self.retrieval_token_idx}).')
    self.lm.resize_token_embeddings(len(tokenizer))

    self.input_embeddings = self.lm.get_input_embeddings()

    print("Restoring pretrained weights for the visual model.")
    if 'clip' in visual_encoder:
      self.visual_model = CLIPVisionModel.from_pretrained(visual_encoder)
    else:
      self.visual_model = AutoModel.from_pretrained(visual_encoder)

    if 'clip' in visual_encoder:
      hidden_size = self.visual_model.config.hidden_size
    else:
      raise NotImplementedError

    if self.args.freeze_vm:
      print("Freezing the VM.")
      self.visual_model.eval()
      for param in self.visual_model.parameters():
        param.requires_grad = False
    else:
      self.visual_model.train()

    self.visual_model_name = visual_encoder

    embedding_dim = self.input_embeddings.embedding_dim * self.args.n_visual_tokens
    self.text_hidden_fcs = nn.ModuleList([])
    if self.args.shared_emb_dim is None:
      if len(self.args.text_emb_layers) == 1:
        if (self.args.text_emb_layers[0] in [-1, self.lm.config.num_hidden_layers]) and ('bert' not in opt_version):
          out_dim = self.lm.config.word_embed_proj_dim
        else:
          out_dim = self.lm.config.hidden_size
      else:
        if (-1 in self.args.text_emb_layers) or (self.lm.config.num_hidden_layers in self.args.text_emb_layers) \
          and (self.lm.config.word_embed_proj_dim != self.lm.config.hidden_size):
          raise ValueError('No projection dim specified but model uses last output layer and an intermediate one (which have different dims).')
        else:
          out_dim = self.lm.config.hidden_size
    else:
      out_dim = self.args.shared_emb_dim

      for layer_idx in self.args.text_emb_layers:
        if (layer_idx == -1 or layer_idx == self.lm.config.num_hidden_layers) and ('bert' not in opt_version):
          in_dim = self.lm.config.word_embed_proj_dim

          text_fc = [nn.Linear(in_dim, out_dim), nn.Dropout(self.args.text_embed_dropout_prob)]
          self.text_hidden_fcs.append(nn.Sequential(*text_fc))

        elif layer_idx < self.lm.config.num_hidden_layers:
          text_fc = [nn.Linear(self.lm.config.hidden_size, out_dim), nn.Dropout(self.args.text_embed_dropout_prob)]
          self.text_hidden_fcs.append(nn.Sequential(*text_fc))
        else:
          raise ValueError(f'Embedding of layer {layer_idx} was requested but model only has {self.lm.config.num_hidden_layers} layers.')

    self.visual_embeddings = nn.Linear(hidden_size, embedding_dim)
    self.visual_fc = nn.Linear(hidden_size, out_dim)

    self.image_dropout = nn.Dropout(self.args.image_embed_dropout_prob)


  def get_visual_embs(self, pixel_values: torch.FloatTensor, mode: str = 'captioning'):
    if mode not in ['captioning', 'retrieval']:
      raise ValueError(f'mode should be one of ["caption", "retrieval"], got {mode} instead.')

    # Extract visual embeddings from the vision encoder.
    if 'clip' in self.visual_model_name:
      outputs = self.visual_model(pixel_values)
      encoder_outputs = outputs.pooler_output
    else:
      raise NotImplementedError

    # Use the correct fc based on function argument.
    if mode == 'captioning':
      visual_embs = self.visual_embeddings(encoder_outputs)  # (2, D * n_visual_tokens)
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], self.args.n_visual_tokens, -1))
    elif mode == 'retrieval':
      visual_embs = self.visual_fc(encoder_outputs)  # (2, D * n_visual_tokens)
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1))
    else:
      raise NotImplementedError

    visual_embs = self.image_dropout(visual_embs)
    return visual_embs


  def train(self, mode=True):
    super(FromageModel, self).train(mode=mode)
    # Overwrite train() to ensure Frozen models remain frozen.
    if self.args.freeze_lm:
      self.lm.eval()
    if self.args.freeze_vm:
      self.visual_model.eval()


  def forward(
    self,
    pixel_values: torch.FloatTensor,
    labels: torch.LongTensor,
    caption_len: torch.LongTensor,
    mode: str = 'captioning',
    concat_captions: bool = False,
    input_prefix: Optional[str] = None,
    inference: bool = False,
  ):
    visual_embs = self.get_visual_embs(pixel_values, mode)

    batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens
    if labels is not None:
      assert labels.shape[0] == batch_size, (visual_embs.shape, labels.shape)

    input_embs = self.input_embeddings(labels)  # (N, T, D)

    last_embedding_idx = caption_len - 1  # -1 to retrieve the token before the eos token

    if input_prefix is not None:
      prompt_ids = self.tokenizer(input_prefix, add_special_tokens=False, return_tensors="pt").input_ids
      prompt_ids = prompt_ids.to(visual_embs.device)
      prompt_embs = self.input_embeddings(prompt_ids)
      prompt_embs = prompt_embs.repeat(batch_size, 1, 1)
      assert prompt_embs.shape[0] == batch_size, prompt_embs.shape
      assert prompt_embs.shape[2] == input_embs.shape[2], prompt_embs.shape
      assert len(prompt_embs.shape) == 3, prompt_embs.shape

    if mode == 'captioning':
      # Concat to text embeddings.
      condition_seq_len = 0
      if input_prefix is None:
        # Just add visual embeddings.
        input_embs = torch.cat([visual_embs, input_embs], axis=1)
        last_embedding_idx += vis_seq_len
        condition_seq_len += vis_seq_len
        full_labels = torch.zeros(visual_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100
      else:
        # Add visual and prompt embeddings.
        prefix_embs = torch.cat([visual_embs, prompt_embs], axis=1)
        input_embs = torch.cat([prefix_embs, input_embs], axis=1)

        last_embedding_idx += prefix_embs.shape[1]
        condition_seq_len += prefix_embs.shape[1]
        full_labels = torch.zeros(prefix_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100

      # Mask out embedding tokens in the labels.
      full_labels = torch.cat([full_labels, labels], axis=1)

      pad_idx = []

      for label in full_labels:
        for k, token in enumerate(label):
          # Mask out retrieval token if it exists.
          if token in [self.tokenizer.pad_token_id, self.retrieval_token_idx]:
            label[k:] = -100
            pad_idx.append(k)
            break
          if k == len(label) - 1:  # No padding found.
            pad_idx.append(k + 1)
      assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

      bs, seq_len, embs_dim = input_embs.shape
      if concat_captions:
        assert len(input_embs.shape) == 3, input_embs
        assert len(full_labels.shape) == 2, full_labels
        assert batch_size % 2 == 0
        all_concat_input_embs = []
        all_concat_labels = []

        # Rearrange embeddings and labels (and their padding) to concatenate captions.
        for i in range(batch_size // 2):
          first_idx = i * 2
          second_idx = first_idx + 1
          first_emb = input_embs[first_idx, :pad_idx[first_idx], :]
          first_labels = full_labels[first_idx, :pad_idx[first_idx]]
          first_padding = input_embs[first_idx, pad_idx[first_idx]:, :]
          first_labels_padding = full_labels[first_idx, pad_idx[first_idx]:]

          second_emb = input_embs[second_idx, :pad_idx[second_idx], :]
          second_labels = full_labels[second_idx, :pad_idx[second_idx]]
          second_padding = input_embs[second_idx, pad_idx[second_idx]:, :]
          second_labels_padding = full_labels[second_idx, pad_idx[second_idx]:]

          assert torch.all(first_labels_padding == -100), first_labels_padding
          assert torch.all(second_labels_padding == -100), second_labels_padding
          concat_input_embs = torch.cat([first_emb, second_emb, first_padding, second_padding], axis=0)   # (T*2, 768)
          concat_labels = torch.cat([first_labels, second_labels, first_labels_padding, second_labels_padding], axis=0)   # (T*2, 768)
          all_concat_input_embs.append(concat_input_embs)
          all_concat_labels.append(concat_labels)

        # Pad to max length.
        input_embs = torch.stack(all_concat_input_embs, axis=0)  # (N/2, T*2, 768)
        full_labels = torch.stack(all_concat_labels, axis=0)  # (N/2, T*2, 768)
        assert input_embs.shape == (bs // 2, seq_len * 2, embs_dim), input_embs.shape
        assert full_labels.shape == (bs // 2, seq_len * 2), full_labels.shape

      output = self.lm(inputs_embeds=input_embs,
                       labels=full_labels,
                       output_hidden_states=True)
    elif mode == 'retrieval':
      full_labels = torch.clone(labels)
      if input_prefix is not None:
        print(f'Adding prefix "{input_prefix}" to retrieval.')
        # Add prompt embeddings.
        prefix_embs = prompt_embs
        input_embs = torch.cat([prefix_embs, input_embs], axis=1)
        last_embedding_idx += prefix_embs.shape[1]
        full_labels = torch.cat([
          torch.zeros(prefix_embs.shape[:2], dtype=torch.int64).to(labels.device) - 100,
          full_labels
        ], axis=1)
      
      pad_idx = []
      for label in full_labels:
        for k, token in enumerate(label):
          if token == self.tokenizer.pad_token_id:
            label[k:] = -100
            pad_idx.append(k)
            break
          if k == len(label) - 1:  # No padding found.
            pad_idx.append(k + 1)
      assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

      output = self.lm(inputs_embeds=input_embs,
                       labels=full_labels,
                       output_hidden_states=True)
    else:
      raise NotImplementedError

    last_embedding = None
    last_output_logit = None
    hidden_states = []

    if mode == 'retrieval':
      if self.args.shared_emb_dim is not None:
        for idx, fc_layer in zip(self.args.text_emb_layers, self.text_hidden_fcs):
          hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, 2048)
      else:
        for idx in self.args.text_emb_layers:
          hidden_states.append(output.hidden_states[idx])

      # Add hidden states together.
      last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

      if not concat_captions:
        last_embedding = torch.stack([last_hidden_state[i, last_embedding_idx[i], :] for i in range(batch_size)], axis=0)  # (N, D)
        last_output_logit = torch.stack([output.logits[i, last_embedding_idx[i] - 1, :] for i in range(batch_size)], axis=0)  # (N, D)
      else:
        raise NotImplementedError

      # Compute retrieval loss.
      assert visual_embs.shape[1] == 1, visual_embs.shape
      visual_embs = visual_embs[:, 0, :]
      visual_embs = visual_embs / visual_embs.norm(dim=1, keepdim=True)
      last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)

      # cosine similarity as logits
      logit_scale = self.logit_scale.exp()
      visual_embs = logit_scale * visual_embs
    elif mode == 'captioning':
      pass
    else:
      raise NotImplementedError

    return output, full_labels, last_embedding, last_output_logit, visual_embs

  def generate(self, embeddings = torch.FloatTensor, max_len: int = 32,
               temperature: float = 0.0, top_p: float = 1.0, min_word_tokens: int = 0,
               ret_scale_factor: float = 1.0, filter_value: float = -float('Inf')):
    """Runs greedy decoding and returns generated captions.

    Args:
      embeddings: Input condition that the model uses for autoregressive generation.
      max_len: Maximum number of tokens to generate.
      temperature: Used to modulate logit distribution.
      top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
      min_word_tokens: Minimum number of words to generate before allowing a [RET] output.
      ret_scale_factor: Proportion to scale [RET] token logits by. A higher value may increase the probability of the model generating [RET] outputs.
      filter_value: Value to assign to tokens that should never be generated.
    Outputs:
      out: (N, T) int32 sequence of output tokens.
      output_embeddings: (N, T, 256) sequence of text output embeddings.
    """
    self.lm.eval()

    with torch.no_grad():  # no tracking history
      batch_size, s, _ = embeddings.shape
      # init output with image tokens
      out = None
      past_key_values = None
      output_embeddings = []
      output_logits = []

      for i in range(max_len):
        if 'opt' in self.opt_version:
          output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
        else:
          if i == 0:
            output = self.lm(inputs_embeds=embeddings, use_cache=True, past_key_values=None, output_hidden_states=True)
          else:
            output = self.lm(input_ids=out[:, -1:], use_cache=True, past_key_values=past_key_values, output_hidden_states=True)

        # Collect and sum the hidden states.
        hidden_states = []
        if self.args.shared_emb_dim is not None:
          for idx, fc_layer in zip(self.args.text_emb_layers, self.text_hidden_fcs):
            hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, 2048)
        else:
          for idx in self.args.text_emb_layers:
            hidden_states.append(output.hidden_states[idx])
        # Add hidden states together.
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)  # (N, T, 256)
        last_embedding = last_hidden_state / last_hidden_state.norm(dim=-1, keepdim=True)
        output_embeddings.append(last_embedding)

        logits = output.logits[:, -1, :]  # (N, vocab_size)
        if top_p == 1.0:
          logits = logits.cpu()
        output_logits.append(logits)

        if self.retrieval_token_idx != -1 and self.retrieval_token_idx is not None:
          if i < min_word_tokens:
            # Eliminate probability of generating [RET] if this is earlier than min_word_tokens.
            logits[:, self.retrieval_token_idx] = filter_value
          else:
            # Multiply by scaling factor.
            logits[:, self.retrieval_token_idx] = logits[:, self.retrieval_token_idx] * ret_scale_factor

        past_key_values = output.past_key_values

        if temperature == 0.0:
          if top_p != 1.0:
            raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
          next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
        else:
          logits = logits / temperature

          # Apply top-p filtering.
          if top_p < 1.0:
            assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # (N, D)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for j in range(sorted_indices.shape[0]):
              indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
              logits[j, indices_to_remove] = filter_value

          token_weights = logits.exp()   # (N, vocab_size)
          next_token = torch.multinomial(token_weights, 1)  # (N, 1)

        next_token = next_token.long().to(embeddings.device)
        if out is not None:
          out = torch.cat([out, next_token], dim=-1)
        else:
          out = next_token

        if 'opt' in self.opt_version:
          next_embedding = self.input_embeddings(next_token)
          embeddings = torch.cat([embeddings, next_embedding], dim=1)
        elif (self.tokenizer.eos_token_id and (next_token == self.tokenizer.eos_token_id).all()):
          # End of generation.
          break

    return out, output_embeddings, output_logits


class Fromage(nn.Module):
  def __init__(self, tokenizer, model_args: Optional[FrozenArgs] = None,
               path_array: Optional[List[str]] = None, emb_matrix: Optional[torch.tensor] = None):
    super().__init__()
    self.model = FromageModel(tokenizer, model_args)
    self.path_array = path_array
    self.emb_matrix = emb_matrix

  def __call__(self, images: Tensor, tgt_tokens: Optional[Tensor] = None, caption_len: Optional[Tensor] = None,
               generate: bool = False, num_words: int = 32, temperature: float = 1.0, top_p: float = 1.0,
               ret_scale_factor: float = 1.0, min_word_tokens: int = 0,
               mode: str = 'captioning', concat_captions: bool = False,
               input_prefix: Optional[str] = None, inference: bool = False) -> Tensor:
    if generate:
      return self.model.generate(images, num_words, temperature=temperature, top_p=top_p,
                                 min_word_tokens=min_word_tokens, ret_scale_factor=ret_scale_factor)
    else:
      output = self.model(
        pixel_values = images,
        labels = tgt_tokens,
        caption_len = caption_len,
        mode = mode,
        concat_captions = concat_captions,
        input_prefix = input_prefix,
        inference = inference)
      return output

  def generate_for_images_and_texts(
    self, prompts: List, num_words: int = 0, ret_scale_factor: float = 1.0, top_p: float = 1.0, temperature: float = 0.0,
    max_num_rets: int = 1, max_img_per_ret: int = 1):
    """
    Encode prompts into embeddings.

    Args:
      prompts: List of interleaved PIL.Image.Image and strings representing input to the model.
      num_words: Maximum number of words to generate for. If num_words = 0, the model will run its forward pass and return the outputs.
      ret_scale_factor: Proportion to scale [RET] token logits by. A higher value may increase the probability of the model generating [RET] outputs.
      top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
      temperature: Used to modulate logit distribution.
      max_num_rets: Maximum number of images to return in one generation pass.
      max_img_per_ret: Maximum number of images to return for each [RET] token.
    Returns:
      return_outputs: List consisting of either str or List[PIL.Image.Image] objects, representing image-text interleaved model outputs.
    """
    input_embs = []
    input_ids = []
    add_bos = True

    for i, p in enumerate(prompts):
      if type(p) == Image.Image:
        # Encode as image.
        pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, p)
        pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
        pixel_values = pixel_values[None, ...]

        visual_embs = self.model.get_visual_embs(pixel_values, mode='captioning')  # (1, n_visual_tokens, D)
        input_embs.append(visual_embs)
      elif type(p) == str:
        text_ids = self.model.tokenizer(p, add_special_tokens=True, return_tensors="pt").input_ids.to(self.model.logit_scale.device)
        if not add_bos:
          # Remove <bos> tag.
          text_ids = text_ids[:, 1:]
        else:
          # Only add <bos> once.
          add_bos = False

        text_embs = self.model.input_embeddings(text_ids)  # (1, T, D)
        input_embs.append(text_embs)
        input_ids.append(text_ids)
      else:
        raise ValueError(f'Input prompts should be either PIL.Image.Image or str types, got {type(p)} instead.')
    input_embs = torch.cat(input_embs, dim=1)
    input_ids = torch.cat(input_ids, dim=1)

    if num_words == 0:
      generated_ids = input_ids
      outputs = self.model.lm(inputs_embeds=input_embs, use_cache=False, output_hidden_states=True)
      # Map outputs to embeddings, so we can retrieve embeddings from the [RET] tokens.
      out = []
      for x, fc in zip(self.model.args.text_emb_layers, self.model.text_hidden_fcs):
          out.append(fc(outputs.hidden_states[x]))
      embeddings = torch.stack(out, dim=-1).sum(dim=-1)
      embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (N, T, 256)
    elif num_words > 0:
      generated_ids, generated_embeddings, _ = self.model.generate(input_embs, num_words,
        temperature=temperature, top_p=top_p, ret_scale_factor=ret_scale_factor)
      embeddings = generated_embeddings[-1][:, input_embs.shape[1]:]

      # Truncate to newline.
      newline_token_id = self.model.tokenizer('\n', add_special_tokens=False).input_ids[0]
      trunc_idx = 0
      for j in range(generated_ids.shape[1]):
        if generated_ids[0, j] == newline_token_id:
          trunc_idx = j
          break
      if trunc_idx > 0:
        generated_ids = generated_ids[:, :trunc_idx]
        embeddings = embeddings[:, :trunc_idx]
    else:
      raise ValueError

    # Save outputs as an interleaved list.
    return_outputs = []
    # Find up to max_num_rets [RET] tokens, and their corresponding scores.
    all_ret_idx = [i for i, x in enumerate(generated_ids[0, :] == self.model.retrieval_token_idx) if x][:max_num_rets]
    seen_image_idx = []  # Avoid showing the same image multiple times.

    last_ret_idx = 0
    if len(all_ret_idx) == 0:
      # No [RET] tokens.
      caption = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
      return_outputs.append(utils.truncate_caption(caption))
    else:
      for ret_idx in all_ret_idx:
        ret_emb = embeddings[:, ret_idx, :]
        scores = self.emb_matrix @ ret_emb.T

        # Downweight seen images.
        for seen_idx in seen_image_idx:
          scores[seen_idx, :] -= 1000

        # Get the top max_img_per_ret + 3 (in case some fail) images for each image.
        _, top_image_idx = scores.squeeze().topk(max_img_per_ret + 3)
        image_outputs = []
        for img_idx in top_image_idx:
          # Find the first image that does not error out.
          try:
            seen_image_idx.append(img_idx)
            img = utils.get_image_from_url(self.path_array[img_idx])
            image_outputs.append(img)
            if len(image_outputs) == max_img_per_ret:
              break
          except UnidentifiedImageError:
            pass

        caption = self.model.tokenizer.batch_decode(generated_ids[:, last_ret_idx:ret_idx], skip_special_tokens=True)[0]
        last_ret_idx = ret_idx + 1
        return_outputs.append(utils.truncate_caption(caption) + ' [RET]')
        return_outputs.append(image_outputs)

    return return_outputs


def load_fromage(model_dir: str) -> Fromage:
  model_args_path = os.path.join(model_dir, 'model_args.json')
  model_ckpt_path = os.path.join(model_dir, 'pretrained_ckpt.pth.tar')
  embs_paths = [s for s in glob.glob(os.path.join(model_dir, 'cc3m_embeddings*.pkl'))]

  if not os.path.exists(model_args_path):
    raise ValueError(f'model_args.json does not exist in {model_dir}.')
  if not os.path.exists(model_ckpt_path):
    raise ValueError(f'pretrained_ckpt.pth.tar does not exist in {model_dir}.')
  if len(embs_paths) == 0:
    raise ValueError(f'cc3m_embeddings_*.pkl files do not exist in {model_dir}.')

  # Load embeddings.
  # Construct embedding matrix for nearest neighbor lookup.
  path_array = []
  emb_matrix = []

  # These were precomputed for all CC3M images with `model.get_visual_embs(image, mode='retrieval')`.
  for p in embs_paths:
    with open(p, 'rb') as wf:
        train_embs_data = pkl.load(wf)
        path_array.extend(train_embs_data['paths'])
        emb_matrix.append(train_embs_data['embeddings'])
  emb_matrix = np.concatenate(emb_matrix, axis=0)

  # Number of paths should be equal to number of embeddings.
  assert len(path_array) == emb_matrix.shape[0], (len(path_array), emb_matrix.shape[0])

  with open(model_args_path, 'r') as f:
      model_kwargs = json.load(f)

  # Initialize tokenizer.
  tokenizer = GPT2Tokenizer.from_pretrained(model_kwargs['opt_version'])
  tokenizer.pad_token = tokenizer.eos_token
  # Add special tokens to the model to enable [RET].
  tokenizer.add_special_tokens({"cls_token": "<|image|>"})
  tokenizer.add_tokens('[RET]')
  ret_token_idx = tokenizer('[RET]', add_special_tokens=False).input_ids
  assert len(ret_token_idx) == 1, ret_token_idx
  model_kwargs['retrieval_token_idx'] = ret_token_idx[0]
  args = namedtuple('args', model_kwargs)(**model_kwargs)

  # Initialize model for inference.
  model = Fromage(tokenizer, args, path_array=path_array, emb_matrix=emb_matrix)
  model = model.eval()
  model = model.bfloat16()
  model = model.cuda()

  # Load pretrained linear mappings and [RET] embeddings.
  checkpoint = torch.load(model_ckpt_path)
  model.load_state_dict(checkpoint['state_dict'], strict=False)
  with torch.no_grad():
      model.model.input_embeddings.weight[model.model.retrieval_token_idx, :].copy_(checkpoint['state_dict']['ret_input_embeddings.weight'].cpu().detach())

  logit_scale = model.model.logit_scale.exp()
  emb_matrix = torch.tensor(emb_matrix, dtype=logit_scale.dtype).to(logit_scale.device)
  emb_matrix = emb_matrix / emb_matrix.norm(dim=1, keepdim=True)
  emb_matrix = logit_scale * emb_matrix
  model.emb_matrix = emb_matrix

  return model

