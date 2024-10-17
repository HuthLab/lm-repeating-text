#!/bin/bash -eux

# Do layer 5 first, since Figs. 2-4 use that layer
# (layers are 0-indexed here)
time for layer in 5 {0..4} {6..11}; do
    time for story in wheretheressmoke fromboyhoodtofatherhood onapproachtopluto eyespy souls; do
        for i in {0..4}; do
            time python3 ./attn_optim.py --bias_method recent_powerlaw --epochs 2000 --output_dir attn-optim-results/l${layer}_${story}_${i} --opt_layers ${layer} --train_story $story
        done
    done
done
