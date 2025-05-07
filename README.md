# RLPreprocessor

RLPreprocessor is a reinforcement learning framework for optimizing photometric image preprocessing (brightness and contrast adjustments) to improve object detection performance using black-box CNN models such as YOLO. It leverages the Deep Deterministic Policy Gradient (DDPG) algorithm to learn continuous transformations that enhance detection confidence and success rates. The system is model-agnostic and particularly applicable to domains like biomedical imaging, where annotation-free tuning is valuable.

## Getting Started

### 1. Set Up the Environment

Clone the repo and install dependencies:

```bash
git clone https://github.com/RFM7991/RL-Preprocessing.git
cd RLPreprocessor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### 2. Prepare Image Data

Add `.jpg` or `.png` images to the following folders:

* `images/train` — used for training the RL agent
* `images/test` — used for evaluation after training

You can use your own dataset or a public object detection dataset (e.g., Ultralytics' examples).

### 3. Train the DDPG Agent

Run the main experiment script:

```bash
python run_experiment.py
```

This will train the DDPG agent to control image brightness and contrast to improve downstream detection. Trained models will be saved to the `models/` directory.

### 4. Evaluate Performance

The system automatically runs evaluation after training using the saved actor/critic models on the test set. Performance plots and logs are saved to `output/DDPG/`.

## Core Concept

The environment models image preprocessing as a continuous control problem. At each step, the agent receives a state vector representing the current photometric properties of the image and outputs actions to adjust brightness (beta) and contrast (alpha). The reward function is based on the improvement in detection confidence or successful detection count when evaluated by a frozen YOLO model. Ground truth annotations are not required.

## Features

* Continuous action space with DDPG
* Model-agnostic design; works with any fixed CNN detector
* Reward based on detection improvement without access to labels
* Plotting and evaluation tools for comparing results across experiments

## Requirements

* Python 3.8+
* PyTorch
* OpenCV
* Matplotlib
* NumPy
* Ultralytics YOLOv8

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Acknowledgments

This project was developed as part of a research effort into reinforcement learning for adaptive image preprocessing. YOLO models are loaded using the Ultralytics framework: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

## Contact

Maintained by Robert McPherson. For questions or collaboration, please reach out to robert.mcpherson@tufts.edu