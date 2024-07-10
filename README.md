# $\beta$-DPO

## What is this repo?

This repo includes a reference implementation of the $\beta$-DPO algorithm for training language models from preference data, as described in the paper `$\beta$-DPO: Direct Preference Optimization with Dynamic $\beta$`

The $\beta$-DPO pipeline has two stages:

1. Run supervised fine-tuning (SFT) on the dataset(s) of interest.
2. Run preference learning on the model from step 1, using preference data (ideally from the same distribution as the SFT examples).

The files in this repo are:
- `train.py`: the main entry point for training (either SFT or $\beta$-DPO preference-based training)
- `trainers.py`: the trainer classes (e.g., implementing the loop of learning as well as multi-GPU logic)
- `utils.py`: some convenience functions used by multiple other files
- `preference_datasets.py`: dataset processing logic for both SFT and $\beta$-DPO preference-based training; **this is where you'll need to make some additions to train on your own data**

## A complete example

Let's work through a complete example training pythia 2.8B on the Anthropic-HH dataset.

### Step 1: Set up environment

First, create a virtualenv and install the dependencies. Python 3.8+ is recommended.
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Step 2: Run SFT

We'll take advantage of FSDP's mixed precision in bfloat16 to speed up training; we usually see about a 50% speedup. By default, SFT will run for a single epoch over a mixture of the selected datasets. Datasets will be downloaded on the fly and cached locally.
```sh
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_beta_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
```
> Note: this command is run on a machine with 4 80GB A100s; on this hardware, SFT takes about 1hr 30min. If you have less compute available, you might need to increase the number of gradient accumulation steps, and SFT will take longer.

### Step 3: Run $\beta$-DPO

Check either wandb (if enabled, it is by default) or your output log to find the local run directory. To run $\beta$-DPO, you'll need the path to the final weights, which will look something like `/some/cache/dir/YOUR_USERNAME/pythia28_hh_sft_bf16_2023-06-21_16-58-17_973996/LATEST/policy.pt`. The `LATEST` directory contains the final set of weights from the end of training.


In the context of $\beta$-DPO, the only requisites are modifying the `filtering ratio` and specifying the `scaling factor`, denoted as `a`. For illustrative purposes, let's consider a `filtering ratio=0.2, a=0.6`  as an example:
```sh
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_beta_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt loss.mode_loss=beta_DPO loss.mode_weight=0.2 loss.a=0.6 
```
On 4 80GB A100s, $\beta$-DPO training took about 2hrs 45min.


## Acknowledgement
The project is built upon [DPO](https://github.com/eric-mitchell/direct-preference-optimization)