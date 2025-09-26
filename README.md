UniWM

we provide full code for training and evaluation here.

## Quick Start

```bash
conda create -n uniwm python=3.10
conda activate uniwm
pip install torch==2.4.0
pip install -r requirements.txt --user
```

## Implementation

### Training

To train the model on multiple datasets, use the following `torchrun` command. This script supports multi-GPU distributed training (we provide an example in train.sh).

```bash
torchrun --nproc_per_node={GPU_NUM_PER_NODE} train.py \
    --model anole \
    --data go_stanford,scand,sacson,recon \
    --data_dir /path/to/your/data_samples \
    --decoder_type anole \
    --image_seq_length 784 \
    --input_format anole \
    --output /path/to/save/output \
    --note {experiment_note} \
    --report_to none \
    --do_train \
    --bfloat16
```

### Evaluation

To evaluate a trained model, use the command below. The script supports several evaluation modes, which can be selected by using the appropriate flag (we provide an example in eval.sh).

``` bash
torchrun --nproc_per_node=<GPU_NUM_PER_NODE> train.py \
    --model anole \
    --model_ckpt /path/to/your/checkpoint \
    --data go_stanford,scand,sacson,recon \
    --data_dir /path/to/your/data_samples \
    --decoder_type anole \
    --image_seq_length 784 \
    --input_format anole \
    --output /path/to/save/eval_results \
    --note {experiment_note} \
    --report_to none \
    \
    # Choose ONE of the following evaluation flags for different eval mode:
    --do_single_step_eval
    # --do_task_level_eval
    # --do_rollout_eval

    # Optional: --use_memory_bank_inference
```
#### Evaluation Flags (choose one):

`--do_single_step_eval`: Evaluates the model's performance on a single step of prediction.

`--do_task_level_eval`: Evaluates the model on the full end-to-end task across an entire trajectory. You can optionally enable the memory bank mechanism by adding the `--use_memory_bank_inference` flag to the command. If this flag is omitted, the evaluation runs with the memory bank disabled.

`--do_rollout_eval`: Generates a full trajectory autoregressively (i.e., the model uses its own previous predictions and ground truth actions as input for the next step) and evaluates the result.



We would like to thank "ANOLE: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation", "Chameleon: Mixed-Modal Early-Fusion Foundation Models", and "Imagine while Reasoning in Space: Multimodal Visualization-of-Thought" for their publicly available codebase, which we referenced during the implementation of Anole training.