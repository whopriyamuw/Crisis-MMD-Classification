# Crisis-MMD-Classification

This repository contains code for classifying crisis-related images using multimodal deep learning approaches.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Crisis-MMD-Classification.git
cd Crisis-MMD-Classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Crisis-MMD-Classification/
├── src/
│   ├── data/           # Data processing and loading utilities
│   ├── models/         # Model architectures and training code
│   └── __init__.py
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Usage

### Training

To train the model:

```bash
python src/models/train.py
```

Here train.py is the train code corresponding to the method.

### Inference

To run inference on trained model:

```bash
python src/models/inference.py
```

Here inference.py is the inference code corresponding to the model trained.

### Vision Language Model

The code for the VLM experiments are located in `notebooks/vlm_instruct.ipynb`. To reproduce the experiments run the Jupyter Notebook in a Google Colab environment with a T4 GPU.