# Humans and language models diverge when predicting repeating text

This repository contains the code and data for our [CoNLL 2023 paper](https://aclanthology.org/2023.conll-1.5/). We show that humans and LMs behave differently in their memorization ability, and we add a learned recency bias to the LMs to make them more similar to humans.

The notebook [`Behavioral analyses.ipynb`](Behavioral analyses.ipynb) contains code to load & visualize the behavioral data, vanilla LM performance, and optimized (with recency bias) performance.

The script [`attn_optim.py`](attn_optim.py) contains the code to optimize an attention bias to match behavioral data. It saves metrics (e.g. corr. with behavioral data, validation loss) during & after training, as well as the learned parameters. You will need a GPT-2 with word-level tokenization, which you can [download here][box-gpt2]. (This takes ~12 minutes to run on a GTX 1080, without much optimization.)

The attention biasing is implemented with a PyTorch forward hook.

[`attn_optim_combos.sh`](attn_optim_combos.sh) will run the optimization script multiple times (random initializations) for every layer and stimulus. (This took several hours on our hardware.)

## License

Code is licensed under the MIT license. Data is licensed under CC Attribution-NonCommercial ([CC-NC](https://creativecommons.org/licenses/by-nc/4.0/)).

## Citing

If you use code or data from this repository, please cite the official CoNLL publication: https://aclanthology.org/2023.conll-1.5/

A version of the paper is also on arXiv: https://arxiv.org/abs/2310.06408

[box-gpt2]: https://utexas.box.com/s/89t8h5za41q7f0utva3zlazh5uce36vl
