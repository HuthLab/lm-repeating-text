#!/usr/bin/env python3
# Attention map optimization
# Optimize over the attention map itself to reach human-like performance

import argparse
import collections
import copy
import itertools
import os
import pathlib
import pickle
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def load_data():
    data = joblib.load('data_aggregate_no-opt.joblib')

    return data

class AttentionBiasModule(torch.nn.Module):
    # Module that initializes & computes attention biases of different forms.
    # Save the output of this module's forward() before you do a forward pass of the LM.
    def __init__(self, bias_method, n_layer, n_head, sparse_rank=None, layerwise_bias=True, num_tokens=None, num_repeats=None):
        super().__init__()
        self.bias_method = bias_method
        self.n_head = n_head
        self.n_layer = n_layer
        self.sparse_rank = sparse_rank

        if layerwise_bias:
            self.learned_attn_bias = torch.zeros(self.n_layer, self.n_head, num_tokens, num_tokens)# * torch.finfo(model.dtype).max
        else:
            self.learned_attn_bias = torch.zeros(1, 1, num_tokens, num_tokens)# * torch.finfo(model.dtype).max

        if bias_method == 'lowrank' or bias_method == 'lowrank+sparse':
            print(f'Learning a low-rank bias (rank {sparse_rank})')
            assert bias_sparsity < min(learned_attn_bias.shape[-2:]), "rank too large!!"
            self.learned_attn_bias_A = torch.zeros(self.learned_attn_bias.shape[:-1] + (sparse_rank,)).to(self.learned_attn_bias)
            self.learned_attn_bias_B = torch.zeros(self.learned_attn_bias.shape[:-2] + (sparse_rank,) + self.learned_attn_bias.shape[-1:]).to(self.learned_attn_bias)
            self.learned_attn_bias_B = torch.randn_like(self.learned_attn_bias_B) # randomly initialize one matrix to make gradients better?
            self.learned_attn_bias_S = torch.zeros_like(self.learned_attn_bias) # sparse component of the attention bias

            self.learned_attn_bias_A = torch.nn.Parameter(self.learned_attn_bias_A)
            self.learned_attn_bias_B = torch.nn.Parameter(self.learned_attn_bias_B)

            if bias_method == 'lowrank+sparse':
                self.learned_attn_bias_S = torch.nn.Parameter(self.learned_attn_bias_S)

        elif bias_method == 'recent':
            recent_k = 10
            #self.learned_attn_bias_recent = torch.triu(torch.tril(torch.ones_like(self.learned_attn_bias), diagonal=-1), diagonal=-recent_k)
            self.learned_attn_bias_scalar = torch.zeros(self.learned_attn_bias.shape[:-2] + (1,1)).to(self.learned_attn_bias)
            self.learned_attn_bias_scalar = torch.nn.Parameter(self.learned_attn_bias_scalar)
            #self.learned_attn_bias_scalar.requires_grad_()

        elif bias_method == 'recent_param':
            # Only change attentions for the past `k` tokens (one parameter per token offset)
            recent_k = span_len
            self.learned_attn_bias_scalar = torch.zeros(self.learned_attn_bias.shape[:-2] + (1,1,recent_k)).to(self.learned_attn_bias)
            self.learned_attn_bias_scalar = torch.nn.Parameter(self.learned_attn_bias_scalar)

        elif bias_method == 'recent_powerlaw':
            # Only change attentions for the past `k` tokens (one parameter per token offset)
            self.learned_attn_bias_base = torch.randn(self.learned_attn_bias.shape[:-2] + (1,1,1)).to(self.learned_attn_bias)#.abs()
            self.learned_attn_bias_decay = torch.randn_like(self.learned_attn_bias_base)
            #self.learned_attn_bias_decay = torch.zeros_like(self.learned_attn_bias_base)
            self.learned_attn_bias_base = torch.nn.Parameter(self.learned_attn_bias_base)
            self.learned_attn_bias_decay = torch.nn.Parameter(self.learned_attn_bias_decay)
            #self.learned_attn_bias_base.requires_grad_()
            #self.learned_attn_bias_decay.requires_grad_()

        elif bias_method == 'past_inst':
            # Only change attentions on the past instances of this token
            self.learned_attn_bias_scalar = torch.zeros(self.learned_attn_bias.shape[:-2] + (1,1,num_repeats-1)).to(self.learned_attn_bias)
            self.learned_attn_bias_insts = torch.stack([torch.diagflat(torch.ones(num_tokens - repeat_idx*span_len, dtype=int), offset=-repeat_idx*span_len) \
                                                for repeat_idx in range(1, num_repeats+1)], dim=-1).to(self.learned_attn_bias)
            self.learned_attn_bias_scalar = torch.nn.Parameter(self.learned_attn_bias_scalar)
            #self.learned_attn_bias_scalar.requires_grad_()

        elif bias_method == 'past_inst_offset':
            # Tokens usually actually pay attention to the token *after* the previous instance of this token.
            past_inst_offset = 1
            # Only change attentions on the past instances of this token
            self.learned_attn_bias_scalar = torch.zeros(self.learned_attn_bias.shape[:-2] + (1,1,num_repeats)).to(self.learned_attn_bias)
            self.learned_attn_bias_scalar = torch.ones_like(self.learned_attn_bias_scalar)
            self.learned_attn_bias_insts = torch.stack([torch.diagflat(torch.ones(num_tokens - repeat_idx*span_len + past_inst_offset, dtype=int), offset=-repeat_idx*span_len + past_inst_offset) \
                                                for repeat_idx in range(1, num_repeats+1)], dim=-1).to(self.learned_attn_bias)
            self.learned_attn_bias_scalar.requires_grad_()
            self.learned_attn_bias_scalar = torch.nn.Parameter(self.learned_attn_bias_scalar)

        else:
            assert bias_method == 'dense'
            self.learned_attn_bias = torch.nn.Parameter(self.learned_attn_bias)


    def forward(self, input_length, num_repeats=None, opt_heads=None):
        #print('input_length', input_length)
        heads_mask = torch.zeros((1,self.learned_attn_bias.shape[1],1,1), dtype=self.learned_attn_bias.dtype, device=self.learned_attn_bias.device)
        if opt_heads is None: opt_heads = list(range(self.n_head))
        heads_mask[:,opt_heads] = 1.

        if self.bias_method == 'lowrank':
            return torch.tril(self.learned_attn_bias_A @ self.learned_attn_bias_B) * heads_mask
        elif self.bias_method == 'lowrank+sparse':
            return torch.tril(self.learned_attn_bias_A @ self.learned_attn_bias_B + self.learned_attn_bias_S) * heads_mask
        elif self.bias_method == 'recent':
            self.learned_attn_bias_recent = torch.triu(torch.tril(torch.ones_like(self.learned_attn_bias), diagonal=-1), diagonal=-recent_k)
            return self.learned_attn_bias_scalar * self.learned_attn_bias_recent * heads_mask
        elif self.bias_method == 'recent_param':
            self.learned_attn_bias_recent = torch.stack([torch.diagflat(torch.ones(self.learned_attn_bias.shape[-1]-k), offset=-k) for k in range(recent_k)], dim=-1).to(self.learned_attn_bias)
            return (self.learned_attn_bias_scalar * self.learned_attn_bias_recent).sum(-1) * heads_mask
        elif self.bias_method == 'recent_powerlaw':
            #print('self.learned_attn_bias_base.shape', self.learned_attn_bias_base.shape)
            """self.learned_attn_bias_recent = torch.stack([torch.diagflat(torch.ones(input_length-k), offset=-k) for k in range(input_length)], dim=-1).to(self.learned_attn_bias)
            self.learned_attn_bias_recent = self.learned_attn_bias_recent.transpose(-1, -2)""";
            powerlaw = (self.learned_attn_bias_base * torch.pow(torch.arange(input_length, device=self.learned_attn_bias_decay.device)+1, -torch.exp(self.learned_attn_bias_decay)))
            #print(powerlaw.shape, self.learned_attn_bias_recent.shape)
            #return (self.learned_attn_bias_base * torch.pow(torch.arange(input_length, device=self.learned_attn_bias_decay.device)+1, -torch.exp(self.learned_attn_bias_decay)) * self.learned_attn_bias_recent).sum(-1) * heads_mask # this is bad
            #return (powerlaw @ self.learned_attn_bias_recent).squeeze(-2) * heads_mask # this is maybe better
            return sum(torch.diag_embed(torch.ones(input_length-k).cuda() * powerlaw.squeeze(-2)[..., k], offset=-k) for k in range(input_length)) # hopefully much more efficient
        elif self.bias_method in ['past_inst', 'past_inst_offset']:
            if self.bias_method == 'past_inst':
                self.learned_attn_bias_insts = torch.stack([torch.diagflat(torch.ones(num_tokens - repeat_idx*span_len, dtype=int), offset=-repeat_idx*span_len) \
                                                       for repeat_idx in range(1, num_repeats+1)], dim=-1).to(self.learned_attn_bias)
            elif self.bias_method == 'past_inst_offset':
                # to delete diagonal, multiply by (torch.finfo(self.learned_attn_bias_scalar.dtype).min)
                self.learned_attn_bias_insts = torch.stack([torch.diagflat(torch.ones(num_tokens - repeat_idx*span_len + past_inst_offset, dtype=int), offset=-repeat_idx*span_len + past_inst_offset) \
                                                       for repeat_idx in range(1, num_repeats+1)], dim=-1).to(self.learned_attn_bias)
            return (self.learned_attn_bias_scalar[..., :num_repeats] * self.learned_attn_bias_insts).sum(-1) * heads_mask
        elif self.bias_method == 'dense':
            return self.learned_attn_bias * heads_mask

        raise ValueError(f"unsupported method {method}")


def compute_metrics(losses, outs, baseline_attns, masks, perplexity=False, learned_bias=True):
    # Compute all metrics for a given epoch (given the outputs of that epoch)
    metrics_record = {}
    metrics_record['kldiv'] = {}
    metrics_record['entropy'] = {}
    metrics_record['attn'] = {k: {} for k in ['curr', 'recent', 'past_inst', 'past_inst_offset', 'first', 'other']} # how much of the attention is being put on other tokens?

    metrics_record['loss'] = losses['loss'].item()
    metrics_record['behav_loss'] = losses['behav'].item()
    if 'val' in losses: metrics_record['val_loss'] = losses['val'].item()
    if 'sparsity' in losses: metrics_record['sparsity_loss'] = losses['sparsity'].item()

    if perplexity:
        metrics_record['corpus_nll'] = np.mean(evaluate_test_perplexity(model, tokenized_test_stories, learned_bias=learned_bias))

    # TODO: look at how closely it approximates the lower triangle of a low-rank matrix. Mirror to make symmetric & look at eigenspectrum?
    for layer_idx in range(model.config.n_layer):
        metrics_record['kldiv'][layer_idx] = torch.tril(torch.nn.functional.kl_div(outs['attentions'][layer_idx].log(), baseline_attns[layer_idx], reduction='none')).mean().item()
        metrics_record['entropy'][layer_idx] = torch.nansum(torch.tril(-outs['attentions'][layer_idx].log() * outs['attentions'][layer_idx]).squeeze(0).mean(0), axis=1).mean().item()

        seen_tokens = torch.zeros_like(outs['attentions'][layer_idx].squeeze(0)[0], dtype=bool) # mask of tokens we've seen so far, so we can compute 'other' easily
        # how much attn on current token?
        metrics_record['attn']['curr'][layer_idx] = outs['attentions'][layer_idx].squeeze(0).diagonal(offset=0, dim1=-2, dim2=-1).mean().item() # average over attn heads & tokens
        seen_tokens += torch.eye(*outs['attentions'][layer_idx].shape[-2:], device=model.device, dtype=bool)
        num_recent_tokens = 5
        metrics_record['attn']['recent'][layer_idx] = torch.triu(torch.tril(outs['attentions'][layer_idx].squeeze(0), diagonal=-1), diagonal=-num_recent_tokens).sum(2).mean().item() # how much attn on past 5 tokens? mean over attn heads & tokens
        seen_tokens += torch.triu(torch.tril(torch.ones_like(outs['attentions'][layer_idx].squeeze(0)[0], dtype=bool), diagonal=-1), diagonal=-num_recent_tokens)
        #print("outs['attentions'][layer_idx].shape", outs['attentions'][layer_idx].shape, "masks['past_inst'].shape", masks['past_inst'].shape)
        metrics_record['attn']['past_inst'][layer_idx] = outs['attentions'][layer_idx].masked_fill(~masks['past_inst'], 0).squeeze(0).sum(2).mean().item() # same token_idx in a previous presentation
        seen_tokens += past_instance_mask
        metrics_record['attn']['past_inst_offset'][layer_idx] = outs['attentions'][layer_idx].masked_fill(~masks['past_inst_offset'], 0).squeeze(0).sum(2).mean().item() # same token_idx in a previous presentation
        seen_tokens += past_instance_mask_offset
        metrics_record['attn']['first'][layer_idx] = outs['attentions'][layer_idx].squeeze(0)[:, :, 0].mean().item()
        seen_tokens[:, 0] = True

        metrics_record['attn']['other'][layer_idx] = outs['attentions'][layer_idx].masked_fill(seen_tokens, 0).squeeze(0).sum(2).mean().item() # sum of all other weights (from this epoch)

    return metrics_record

def evaluate_test_perplexity(model, tokenized_test_stories: List[torch.Tensor], learned_bias=False):
    if learned_bias: # oh this is so hacky
        # Are we using a learned bias yet? (for baseline metrics, we do not)
        global learned_attn_bias
    # Evaluate raw language modelling performance on a set of stories
    # Actually returns nll
    nlls = []
    for tokens in tqdm(tokenized_test_stories, desc='stories (perplexity)', leave=False):
        story_token_nll = [] # ppx for each token

        with torch.no_grad():
            batch_input_tokens = tokens[:model.config.n_ctx]
            batch_target_tokens = tokens[1:model.config.n_ctx]
            if learned_bias:
                # shape of the learned bias is dependent on the length of the input sequence
                learned_attn_bias = attn_bias_module(input_length=batch_input_tokens.shape[0], num_repeats=0)
            outs = model(batch_input_tokens)
            batch_nll = torch.nn.functional.cross_entropy(outs['logits'][:-1, :], batch_target_tokens, reduction='none').squeeze(0)
            story_token_nll.append(batch_nll)

            # Iterate one token at a time for any tokens beyond the model's context length (probably 1024)
            for start_idx in tqdm(range(1, tokens.shape[0]-model.config.n_ctx+1), desc='batches', leave=False):
                end_idx = start_idx + model.config.n_ctx
                assert end_idx <= tokens.shape[0], f"out of bounds! {end_idx} > {tokens.shape[0]}"

                batch_input_tokens = tokens[start_idx:end_idx]
                batch_target_tokens = tokens[start_idx+1:end_idx]
                if learned_bias and start_idx == 1:
                    learned_attn_bias = attn_bias_module(input_length=batch_input_tokens.shape[0], num_repeats=0)
                outs = model(batch_input_tokens)
                batch_nll = torch.nn.functional.cross_entropy(outs['logits'][:-1, :], batch_target_tokens, reduction='none').squeeze(0)
                story_token_nll.append(batch_nll[-1:])

        nlls.append(torch.cat(story_token_nll).mean().cpu().numpy())
    return nlls # return average NLL within each story

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize attention map to match LM and human performance')
    parser.add_argument('--bias_method', type=str, required=True,
                        help='Attention map form to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to "train" the attention bias')
    parser.add_argument('--train_story', type=str, default='wheretheressmoke',
                        help='Train attention bias on this story')
    parser.add_argument('--output_dir', type=pathlib.Path, default=pathlib.Path('attn-optim-results/output'),
                        help='Directory to save results')
    parser.add_argument('--opt_layers', type=int, nargs='+', default=[5],
                        help='Set a fixed random seed')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set a fixed random seed')
    parser.add_argument('--no_corpus_ppx', action='store_false', dest='corpus_ppx',
                        help='Set a fixed random seed')
    args = parser.parse_args()

    if args.seed is not None:
        print('Set seed:', args.seed)
        np.random.seed(seed=args.seed)
        torch.manual_seed(args.seed)

    # Load behavioral data from Gorilla
    behavioral_data = load_data()
    story_avg_accs, story_prompt_words, story_span_lens, story_num_repeats, story_words = (behavioral_data[x] for x in ['story_avg_accs', 'story_prompt_words', 'story_span_lens', 'story_num_repeats', 'story_words'])
    all_stories = list(story_span_lens.keys())

    ## Optimizing the attention map
    device = torch.device('cuda')
    ckpt_path = 'gpt2_wordlevel'
    tok_path = ckpt_path

    print('Available stimuli for optimization:', all_stories)


    # Set up model & optimization config
    model = GPT2LMHeadModel.from_pretrained(ckpt_path).to(device)
    model.eval();

    story = args.train_story
    span_len = story_span_lens[story]
    num_repeats = story_num_repeats[story]
    num_tokens = len(story_words[story])
    layerwise_bias = True # True or False; optimize individual biases for all heads & layers
    #layerwise_bias = False
    #opt_layers = list(range(model.config.n_layer)) # which layers to optimize?
    opt_layers = args.opt_layers
    opt_heads = list(range(model.config.n_head)) # which heads to optimize?
    #opt_heads = list(set(opt_heads).difference([1,9]))
    #opt_heads = [1]

    # Proportion of dataset to include in training split. All heldout prompts are evenly allocated between presentations
    train_split_prop = 0.7

    with open(os.path.join(tok_path, 'token_dict.pkl'), 'rb') as f:
        token_dict = pickle.load(f)
        word2int = token_dict['word2int']
        int2word = token_dict['int2word']
        unk_token = token_dict['UNK']

    # Tokenization function
    def encode_wordlevel(word_list: List[str]):
        # TODO: use Tokenizers library, like so:
        # https://github.com/huggingface/tokenizers/issues/244
        # WARNING: this uses global variables!
        return [word2int[w] if w in word2int else word2int[unk_token] for w in word_list]

    # Load stories for perplexity evaluation
    #story_tokens = pickle.load(open('data/stimulidb.pickle', 'rb'))
    story_tokens = joblib.load('perplexity_eval_stimuli.joblib')
    test_stories = ['wheretheressmoke', 'fromboyhoodtofatherhood', 'onapproachtopluto', 'tosailonanaliensea', 'grandmothersmalpaandmyrtle', 'thisisgoingtosuck', 'learninghumanityfromdogs', 'afightingchance', 'thehuntergathererparkingdivision', 'akissdeferred', 'aforgottenprayer']
    test_stories = set(test_stories) - set(all_stories) # don't evaluate on stories we're optimizing with
    assert all((story in story_tokens) for story in test_stories)
    tokenized_test_stories = [torch.tensor(encode_wordlevel(story_tokens[story].split())).to(model.device) for story in test_stories]


    ## Set up data and metrics

    # Top-1 accuracy for each prompt, averaged across all participants
    subject_acc = torch.from_numpy(story_avg_accs[story]['is_correct'].to_numpy()).to(model.device).to(model.dtype)
    # the # of subjects that were given each prompt
    num_subjects = torch.from_numpy(story_avg_accs[story]['num_subjects'].to_numpy()).to(model.device)

    # Hold out subset of prompts in the training story to compute within-story
    # generalization
    train_idxs = story_avg_accs[story].groupby('repeat_idx').sample(frac=train_split_prop, replace=False).index # uniformly allocate val. set across presentations
    train_mask = story_avg_accs[story]['is_correct'].index.isin(train_idxs)

    # Create a mask of the attn. mtx. to calculate how much attn. is put on previous instances of the same token.
    # Creates a banded matrix, where bands are spaced with an interval of `span_len`
    past_instance_mask = torch.zeros((num_tokens, num_tokens), dtype=int)
    past_instance_mask_offset = torch.zeros((num_tokens, num_tokens), dtype=int) # mask for induction head behavior
    for repeat_idx in range(1,num_repeats+1):
        past_instance_mask += torch.diagflat(torch.ones(num_tokens - repeat_idx*span_len, dtype=int), offset=-repeat_idx*span_len)
        past_instance_mask_offset += torch.diagflat(torch.ones(num_tokens - repeat_idx*span_len+1, dtype=int), offset=-repeat_idx*span_len+1)
    past_instance_mask = past_instance_mask.bool().cuda()
    past_instance_mask_offset = past_instance_mask_offset.bool().cuda()


    def compute_metrics_and_losses(outs, baseline_attns, bias_method=None, perplexity=False):
        # This uses several local vars (like `story_prompt_words`, `tokenized`)
        target_token_idxs = story_prompt_words[story].index.astype(int)
        target_token_logits = outs['logits'][target_token_idxs-1, :]
        target_token_id = tokenized[target_token_idxs]
        target_nll = torch.nn.functional.cross_entropy(target_token_logits, target_token_id, reduction='none').squeeze(0)
        nll = torch.nn.functional.cross_entropy(outs['logits'][:-1, :], tokenized[1:], reduction='none').squeeze(0)

        behav_loss = (subject_acc - torch.exp(-target_nll)) * torch.sqrt(num_subjects)
        val_loss = behav_loss[~train_mask].clone().square().mean()
        behav_loss = behav_loss[train_mask].square().mean()
        loss = behav_loss.clone()
        if bias_method == 'lowrank+sparse':
            sparsity_loss = sparsity_penality * learned_attn_bias_S.norm(p=1, dim=-1).max(-1).values.mean()
            loss += sparsity_loss

        losses = {'loss': loss, 'behav': behav_loss, 'val': val_loss}

        # Get metrics
        with torch.no_grad():
            if bias_method == 'lowrank+sparse': losses['sparsity'] = sparsity_loss
            # Metrics for this epoch (or evaluation, etc.)
            this_metrics = compute_metrics(losses, outs, baseline_attns, masks={'past_inst': past_instance_mask, 'past_inst_offset': past_instance_mask_offset}, perplexity=perplexity, learned_bias=(bias_method is not None))

            model_acc_df = story_avg_accs[story][['word_idx', 'repeat_idx', 'is_correct', 'word']].copy()
            model_acc_df['acc'] = torch.exp(-target_nll).cpu().numpy() # model accuracy
            this_metrics['acc_corr_rep0'] = model_acc_df[model_acc_df['repeat_idx'] == 0][['is_correct', 'acc']].corr().iloc[0,1]
            this_metrics['acc_corr_rep1+'] = model_acc_df[model_acc_df['repeat_idx'] > 0][['is_correct', 'acc']].corr().iloc[0,1]

        return {'losses': losses, 'target_nll': target_nll,
                'nll': nll, 'metrics': this_metrics}

    # Get the actual baselines / metrics
    tokenized = torch.tensor(encode_wordlevel(story_words[story])).to(model.device)
    with torch.no_grad():
        outs_baseline = model(tokenized, output_hidden_states=True, output_attentions=True)
        baseline_attns = outs_baseline['attentions']

        metrics_and_losses = compute_metrics_and_losses(outs_baseline, baseline_attns, perplexity=args.corpus_ppx)
        baseline_target_nll, baseline_nll, losses_baseline = (metrics_and_losses[k] for k in ['target_nll', 'nll', 'losses'])
        baseline_loss = losses_baseline['loss']
        metrics_baseline = metrics_and_losses['metrics']
        if 'corpus_nll' in metrics_baseline: print('Baseline corpus nll:', metrics_baseline['corpus_nll'])

    print("Baseline loss:", baseline_loss.item())
    print('Baseline val. loss:', losses_baseline['val'].item())

    assert (torch.tril(baseline_attns[0]) == baseline_attns[0]).all(), 'causal attention map should be LOWER triangular!'

    # this should modify the _inputs_ to the attention (i.e. change attn mask before attn is computed)
    # BUT this relies on HF Transformers converting the mask from bools --> floats _before_ the attention module is called
    # See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L205
    def attn_pre_hook(module, args, kwargs, layer_idx=None, layerwise=False):
        attention_mask = kwargs['attention_mask']
        if layerwise:
            kwargs['attention_mask'] = torch.tril(learned_attn_bias[layer_idx])# * baseline_presoft_attn_mean[layer_idx]
        else:
            kwargs['attention_mask'] = torch.tril(learned_attn_bias)# * baseline_presoft_attn_mean[layer_idx]
        return (args, kwargs)

    hooks_pre = []
    for layer_idx, gpt_layer in enumerate(model.transformer.h):
        if layer_idx not in opt_layers: continue # only create hooks for layers that we're using to optimize the mask
        hook_pre = gpt_layer.attn.register_forward_pre_hook((lambda layer_idx: (lambda *args: attn_pre_hook(*args, layer_idx=layer_idx, layerwise=layerwise_bias)))(layer_idx), with_kwargs=True)
        hooks_pre.append(hook_pre)


    ### Optimization loop

    bias_method = args.bias_method


    ## INITIALIZATION

    bias_sparsity = 1 # if None, then dense. Otherwise, set to the rank of the sparse matrix
    #sparse_bias = (bias_sparsity is not None)
    sparsity_penality = 5e-4

    attn_bias_module = AttentionBiasModule(bias_method, model.config.n_layer, model.config.n_head,
                                           num_tokens=tokenized.shape[0], num_repeats=num_repeats)
    attn_bias_module = attn_bias_module.to(model.device)
    learned_attn_bias = attn_bias_module(tokenized.shape[0], num_repeats=num_repeats)

    with torch.no_grad():
        learned_attn_bias_init = learned_attn_bias.clone().detach()

    opt = torch.optim.Adam(attn_bias_module.parameters(), lr=5e-3) # default lr=1e-3 works pretty well, but slow
    print('Number of learnable parameters:', sum(x.nelement() for x in opt.param_groups[0]['params']))

    ## Optimization

    num_epochs = args.epochs
    metrics = []
    for epoch_idx in tqdm(range(num_epochs), desc='num_epochs'):
        learned_attn_bias = attn_bias_module(input_length=tokenized.shape[0], num_repeats=num_repeats)
        outs = model(tokenized, output_hidden_states=True, output_attentions=True)

        metrics_and_losses = compute_metrics_and_losses(outs, outs_baseline['attentions'], bias_method=bias_method)
        target_nll, nll, losses = (metrics_and_losses[k] for k in ['target_nll', 'nll', 'losses'])
        metrics.append(metrics_and_losses['metrics'])
        loss = losses['loss']

        opt.zero_grad()
        loss.backward()
        opt.step()

    assert not (~torch.isfinite(learned_attn_bias)).any()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recompute final outputs, but without gradients
    with torch.no_grad():
        learned_attn_bias = attn_bias_module(input_length=tokenized.shape[0], num_repeats=num_repeats)
        outs_final = model(tokenized, output_hidden_states=True, output_attentions=True)

        metrics_and_losses = compute_metrics_and_losses(outs_final, baseline_attns, bias_method=bias_method, perplexity=args.corpus_ppx)
        target_nll, nll, losses_final = (metrics_and_losses[k] for k in ['target_nll', 'nll', 'losses'])
        loss = losses_final['loss']
        metrics_final = metrics_and_losses['metrics']
        if 'corpus_nll' in metrics_final: print('Final corpus nll:', metrics_final['corpus_nll'])

    perf_df = story_avg_accs[story][['word_idx', 'repeat_idx', 'is_correct', 'word']].copy()
    perf_df['baseline_acc'] = torch.exp(-baseline_target_nll).cpu().numpy()
    perf_df['final_acc'] = torch.exp(-target_nll).cpu().numpy()
    perf_df['train_mask'] = train_mask

    # This is in the order of ~20 MB
    #print('attention output size in MB:', sum(x.nelement() for x in outs['attentions']) * 4 / (1<<20))

    print(f"Change in training loss: {losses_baseline['loss'].item():.3E} {losses_final['loss'].item():.3E}")
    print(f"Change in training loss (rel.)  : {(losses_final['loss'].item() - losses_baseline['loss'].item()) / losses_baseline['loss'].item():.3f}")

    print(f"Change in validation loss: {losses_baseline['val'].item():.3E} {losses_final['val'].item():.3E}")
    print(f"Change in validation loss (rel.): {(losses_final['val'].item() - losses_baseline['val'].item()) / losses_baseline['val'].item():.3f}")

    print(f"Change in overall perplexity: {nll.mean().exp():.2f} {baseline_nll.mean().exp():.2f}")
    print(f"Change in overall perplexity (rel.): {(nll - baseline_nll).mean().exp().item() - 1:.3f}")

    baseline_attn_df = pd.DataFrame.from_dict(metrics_baseline['attn']).rename_axis('layer_idx', axis=0).rename_axis('attn_metric', axis=1)
    final_attn_df = pd.DataFrame.from_dict(metrics_final['attn']).rename_axis('layer_idx', axis=0).rename_axis('attn_metric', axis=1)


    kldiv_by_token = {layer_idx: torch.tril(torch.nn.functional.kl_div(outs_final['attentions'][layer_idx].log(), baseline_attns[layer_idx], reduction='none')).mean(3).mean(1).squeeze(0).cpu().numpy() \
                            for layer_idx in range(model.config.n_layer)}
    kldiv_by_token_df = pd.DataFrame.from_dict(kldiv_by_token)
    kldiv_by_token_df.index.name = 'token_idx'
    kldiv_by_token_df.columns.name = 'layer_idx'

    # All data (incl. 1st presentation)
    print('Corr over all data:', perf_df[perf_df['repeat_idx'] == 0][['is_correct', 'baseline_acc', 'final_acc']].corr().loc['is_correct', ['baseline_acc', 'final_acc']])
    # Only after the 1st presentation
    print('Corr after 1st presentation:', perf_df[perf_df['repeat_idx'] > 0][['is_correct', 'baseline_acc', 'final_acc']].corr().loc['is_correct', ['baseline_acc', 'final_acc']])
    print('average kldiv by layer:', kldiv_by_token_df.mean(axis=0))

    ## Plotting transfer performance
    # from train story --> other stories
    training_metrics = {'training': copy.deepcopy(metrics), 'baseline': copy.deepcopy(metrics_baseline), 'final': copy.deepcopy(metrics_final)}
    training_metrics['baseline']['attn_maps'] = [] # add baseline & final attention maps
    training_metrics['final']['attn_maps'] = []
    for layer_idx, (layer_attn_baseline, layer_attn_final) in enumerate(zip(outs_baseline['attentions'], outs_final['attentions'])):
        training_metrics['baseline']['attn_maps'].append(layer_attn_baseline.log().squeeze(0).cpu().numpy())
        training_metrics['final']['attn_maps'].append(layer_attn_final.log().squeeze(0).cpu().numpy())

    # Evaluate parameters on all other stories
    training_perf_df = perf_df.copy(deep=True) # we might overwrite the name `perf_df` later
    all_story_metrics = {}
    all_story_perf_df = {} # heldout performance by word in every story (in a nice dataframe)
    for story in tqdm(story_words, desc='heldout stories'):
        if story == args.train_story: continue # the training story is not part of the heldout set!

        span_len = story_span_lens[story]
        num_repeats = story_num_repeats[story]
        num_tokens = len(story_words[story])

        tokenized = torch.tensor(encode_wordlevel(story_words[story])).to(model.device)
        subject_acc = torch.from_numpy(story_avg_accs[story]['is_correct'].to_numpy()).to(model.device).to(model.dtype)
        num_subjects = torch.from_numpy(story_avg_accs[story]['num_subjects'].to_numpy()).to(model.device)

        # These don't really matter
        train_idxs = story_avg_accs[story].groupby('repeat_idx').sample(frac=train_split_prop, replace=False).index # uniformly allocate val. set across presentations
        train_mask = story_avg_accs[story]['is_correct'].index.isin(train_idxs)

        # Need to remake these matrices for each story b/c they have different
        # span lengths & num. of presentations
        past_instance_mask = torch.zeros((num_tokens, num_tokens), dtype=int)
        past_instance_mask_offset = torch.zeros((num_tokens, num_tokens), dtype=int)
        for repeat_idx in range(1,num_repeats+1):
            past_instance_mask += torch.diagflat(torch.ones(num_tokens - repeat_idx*span_len, dtype=int), offset=-repeat_idx*span_len)
            past_instance_mask_offset += torch.diagflat(torch.ones(num_tokens - repeat_idx*span_len+1, dtype=int), offset=-repeat_idx*span_len+1)
        past_instance_mask = past_instance_mask.bool().cuda()
        past_instance_mask_offset = past_instance_mask_offset.bool().cuda()

        with torch.no_grad():
            # Use the module once to get the correct sizes, then overwrite with
            # zeros
            learned_attn_bias = attn_bias_module(input_length=tokenized.shape[0], num_repeats=story_num_repeats[story])
            learned_attn_bias = torch.zeros_like(learned_attn_bias)
            outs_baseline = model(tokenized, output_hidden_states=True, output_attentions=True)
            baseline_attns = outs_baseline['attentions']

            metrics_and_losses = compute_metrics_and_losses(outs_baseline, baseline_attns)
            baseline_target_nll, baseline_nll, losses_baseline = (metrics_and_losses[k] for k in ['target_nll', 'nll', 'losses'])
            baseline_loss = losses_baseline['loss']
            metrics_baseline = metrics_and_losses['metrics']

        # Use the true value of the attention bias to get post-optimization performance on the held-out story
        with torch.no_grad():
            learned_attn_bias = attn_bias_module(input_length=tokenized.shape[0], num_repeats=story_num_repeats[story])
            outs_final = model(tokenized, output_hidden_states=True, output_attentions=True)

            metrics_and_losses = compute_metrics_and_losses(outs_final, baseline_attns, bias_method=bias_method)
            target_nll, nll, losses_final = (metrics_and_losses[k] for k in ['target_nll', 'nll', 'losses'])
            loss = losses_final['loss']
            metrics_final = metrics_and_losses['metrics']

        perf_df = story_avg_accs[story][['word_idx', 'repeat_idx', 'is_correct', 'word']].copy()
        perf_df['baseline_acc'] = torch.exp(-baseline_target_nll).cpu().numpy()
        perf_df['final_acc'] = torch.exp(-target_nll).cpu().numpy()

        # Save values for this held-out story
        all_story_metrics[story] = {'baseline': metrics_baseline, 'final': metrics_final, 'attn_change': [], 'perf_df': perf_df}

        for layer_idx, (layer_attn_baseline, layer_attn_final) in enumerate(zip(outs_baseline['attentions'], outs_final['attentions'])):
            attn_matrix_change = (layer_attn_final.log() - layer_attn_baseline.log()).squeeze(0).cpu().numpy()
            all_story_metrics[story]['attn_change'].append(attn_matrix_change)

    # Save the relevant outputs so we can aggregate them
    heldout_metrics_fname = output_dir / 'heldout_metrics.joblib'
    print('Saving held-out results to:', heldout_metrics_fname)
    joblib.dump(all_story_metrics, open(heldout_metrics_fname, 'wb'))

    training_metrics_fname = output_dir / 'training_metrics.joblib'
    print('Saving training results to:', training_metrics_fname)
    joblib.dump(training_metrics, open(training_metrics_fname, 'wb'))

    attn_bias_module = attn_bias_module.cpu()
    bias_module_fname = output_dir / 'bias_params.pt'
    print('Saving learned bias parameters to:', bias_module_fname)
    torch.save(attn_bias_module.state_dict(), bias_module_fname)

    perf_df_fname = output_dir / 'training_perf_df.pkl'
    print('Saving training performance df to:', perf_df_fname)
    training_perf_df.to_pickle(perf_df_fname)
