# Grounding Language Models to Images for Multimodal Generation

![FROMAGe model](./teaser.png)

This repository hosts the code and model weights for FROMAGe.

[Paper](https://arxiv.org/abs/2301.13823) | [Project Webpage](https://jykoh.com/fromage)


## Setup instructions

### Environment
Set up a new virtualenv, and install required libraries:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add the `fromage` library to PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/home/path/to/fromage/
```

### Pretrained Checkpoints

The FROMAGe model weights (linear layers and [RET] embedding) are small (around 11MB), and are included in this Git repo. They will be in the `fromage_model/` folder after cloning. The checkpoint and model config in `fromage_model/` reproduce the results reported in our paper.

We have also included a second model trained with a stronger visual linear layer (4 visual tokens instead of 1), located at `fromage_model/fromage_vis4`. This model generally does better on dialogue settings and does not require as much tuning of inference time hyperparameters, as it is able to better represent more complex images.

### Precomputed Embeddings For Image Retrieval

The visual embeddings for Conceptual Captions images with valid URLs are precomputed and stored at this [URL](https://drive.google.com/file/d/1wMojZNqEwApNlsCZVvSgQVtZLgbeLoKi/view?usp=share_link). These are used to enable the model to retrieve images. The embeddings take up around 3GB, and are compatible with both model configs we provide. Download the files and place `cc3m_embeddings.pkl` into the `fromage_model/` directory.


## Inference

Check out `FROMAGe_example_notebook.ipynb` for examples on calling the model for inference. Several of the figures presented in the paper are reproduced in this notebook using greedy decoding of the model. Note that there may be minor differences in image outputs due to CC3M images being lost over time.


## Training

Coming soon!


## TODOs

- [ ] Add training code and instructions for training a new model on CC3M.
- [ ] Implement [LLM.int8()](https://arxiv.org/abs/2208.07339) for inference with lower memory GPUs.



## Citation

If you find this work useful, please consider citing:

```
@inproceedings{koh2023grounding,
  title={Grounding Language Models to Images for Multimodal Generation},
  author={Koh, Jing Yu and Salakhutdinov, Ruslan and Fried, Daniel},
  journal={arXiv:2301.13823},
  year={2023}
}
```