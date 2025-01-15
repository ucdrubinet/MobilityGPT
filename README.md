# MobilityGPT

A road-segment-level GPT model for generating and predicting mobility trajectories.

## Overview

MobilityGPT learns to generate realistic mobility trajectories by training on road-segment-level movement data. The model combines transformer architecture with spatial awareness to generate high-quality trajectory predictions.

### Key Features
- Road network awareness through adjacency matrices
- Gravity-based trajectory sampling
- LoRA (Low-Rank Adaptation) fine-tuning support
- Differential Privacy training options
- Multiple training approaches (Base, DPO, PPO)


### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Required Files
Your dataset directory (e.g., `SF-Taxi/`) should contain:
- `roadmap.geo`: Road segment geographical information
- `roadmap.rel`: Road segment connectivity information
- Trajectory data file with comma-separated road segment IDs

## Configuration

### Main Parameters
- `dataset`: Dataset name (e.g., "SF")
- `validation_split`: Train/validation split ratio (default: 0.2)
- `eps`: Epsilon value for differential privacy
- `num_samples`: Number of synthetic trajectories to generate

### Model Configuration
- `block_size`: Maximum trajectory length
- `learning_rate`: Training learning rate
- `lora_rank`: Rank for LoRA adaptation
- `lora_alpha`: Alpha value for LoRA
- `lora_dropout`: Dropout rate for LoRA

## Model Features

### Base Model
- Road segment-level GPT architecture
- Spatial awareness through adjacency matrices
- Gravity-based sampling support

### Training Options
- Standard training
- LoRA fine-tuning
- Differential Privacy
- PPO fine-tuning
- DPO training


## Citation

If you find this code useful in your research, please consider citing:

```
@article{mobilitygpt,
  title={Mobilitygpt: Enhanced human mobility modeling with a gpt model},
  author={Haydari, Ammar and Chen, Dongjie and Lai, Zhengfeng and Zhang, Michael and Chuah, Chen-Nee},
  journal={arXiv preprint arXiv:2402.03264},
  year={2024}
}
```
