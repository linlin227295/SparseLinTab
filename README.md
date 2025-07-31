# SparseLinTab:Sparse Linear Self-Attention for Efficient Feature Interaction in Tabular Data

  **[Overview](#overview)**
| **[Abstract](#abstract)**
| **[Installation](#installation)**
| **[Examples](#examples)**
| **[Citation](#citation)**


## Overview

Hi, good to see you here! 👋

Thanks for checking out the code for SparseLinTab.

This codebase will allow you to reproduce experiments from the paper as well as use SparseLinTab for your own research.

## Abstract

Tabular data supports intelligent decision-making in key areas such as finance and healthcare, but its heterogeneous feature interaction modeling and large-scale computing efficiency issues have long restricted the application of deep learning technology. This paper proposes SparseLinTab, an efficient tabular data modeling framework based on bidirectional sparse linear self-attention. By decoupling the interactions between rows (sample level) and columns (feature level), SparseLinTab reduces the quadratic computational complexity of traditional Transformer self-attention $O(N^{2}M + NM^{2})$ to linear $O(NM)$, while retaining the ability to model global dependencies. Experiments verify its superiority in 7 public datasets (covering classification and regression tasks), and the average performance of SparseLinTab is better than all traditional gradient boosting models and deep learning models. In addition, ablation experiments and attention visualization show that the sparse mechanism significantly enhances the robustness and generalization of the model by filtering noise interactions and focusing on key feature combinations.

## Installation

Set up and activate the Python environment by executing

```
conda env create -f environment.yml
conda activate SparseLinTab
```
 
If you are running this on a system without a GPU, use the above with `environment_no_gpu.yml` instead.

## Examples

We now give some basic examples of running SparseLinTab.

SparseLinTab downloads all supported datasets automatically, so you don't need to worry about that.

We use [wandb](http://wandb.com/) to log experimental results.
Wandb allows us to conveniently track run progress online.
If you do not want wandb enabled, you can run `wandb off` in the shell where you execute SparseLinTab.

For example, run this to explore SparseLinTab with default configuration on Higgs

```
python run.py --data_set higgs --project higgs-default --exp_name DLCFormer --dimk_row=128 --dimk_col=8 --data_loader_nprocs=4 --exp_eval_every_n 1 --exp_eval_every_epoch_or_steps epochs --exp_optimizer_warmup_proportion 0.7 --exp_optimizer lookahead_lamb --exp_azure_sweep False --metrics_auroc True --exp_print_every_nth_forward 100 --model_num_heads 8 --model_stacking_depth 8 --exp_lr 1e-3 --exp_scheduler flat_and_anneal --exp_num_total_steps 500000 --exp_gradient_clipping 1 --exp_batch_size 4096 --model_dim_hidden 64 --model_augmentation_bert_mask_prob "dict(train=0.15, val=0, test=0)" --exp_tradeoff 1 --exp_tradeoff_annealing cosine --model_checkpoint_key higgs__bs_4096__feature_mask_DLCFormer --exp_device cuda:3
```

Another example: A run on the Income dataset may look like this

```
python run.py --data_set income --project income-default --exp_name DLCFormer --dimk_row=128 --dimk_col=16 --data_loader_nprocs=2 --exp_eval_every_n 5 --exp_eval_every_epoch_or_steps epochs --exp_optimizer_warmup_proportion 0.7 --exp_optimizer lookahead_lamb --exp_azure_sweep False --metrics_auroc True --exp_print_every_nth_forward 100 --model_num_heads 8 --model_stacking_depth 8 --exp_lr 1e-3 --exp_scheduler flat_and_anneal --exp_num_total_steps 2000000 --exp_gradient_clipping 1 --exp_batch_size 2048 --model_dim_hidden 64 --model_augmentation_bert_mask_prob "dict(train=0.0, val=0.0, test=0.0)" --exp_tradeoff 0 --exp_tradeoff_annealing constant --model_label_bert_mask_prob "dict(train=0.15, val=0.0, test=0.0)" --exp_cache_cadence 1 --model_checkpoint_key income__bs_2048__label_mask_DLCFormer --exp_device cuda:0
```

You can find all possible config arguments and descriptions in `SparseLinTab/configs.py` or using `python run.py --help`.

In `scripts/` we provide a list with the runs and correct hyperparameter configurations presented in the paper.

We hope you enjoy using the code and please feel free to reach out with any questions 😊


## Citation

If you find this code helpful for your work, please cite our paper.