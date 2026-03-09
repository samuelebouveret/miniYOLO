# MiniYOLO

## Project Overview

MiniYOLO is a lightweight implementation of the YOLOv1 (You Only Look Once) object detection model, designed for efficient training and inference on small datasets and resource-constrained environments. Significant changes are performed on the architecture of the model by diminishing the number of filters and layers, the image size is set to *(88, 88)*, and the YOLO specific parameters are hardcoded (S=2 B=2 C=3). 

## Installation

1. **Clone the repository:**
	```bash
	git clone <repo-url>
	cd miniYOLO
	```

2. **Create and activate a Python virtual environment (recommended):**
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```

3. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```

**Python version:** The project was developed on  **Python 3.13**.

## Folder Structure

```
MiniYOLO/
├── miniyolo.py            # Main training script for MiniYOLO
├── requirements.txt       # Python dependencies
├── data/                  # Dataset folder (see below)
│   ├── images/             # VOC images (JPEG)
│   └── annotations/        # VOC annotations (XML)
├── model/                 # Model architecture, loss, and callbacks
│   ├── __init__.py
│   ├── model_miniyolo.py   # Model definition and utils
│   └── loss_miniyolo.py    # Loss function
│
├── trained/               # Output directory for trained models and weights
│   ├── models/             # Saved Keras models
│   ├── weights/            # Saved weights
│   └── loss_curve.png      # Val_loss/loss per epoch plot
└── README.md              # Project documentation
```


## Dataset Preparation

The `data/` directory should contain images and annotations in the PASCAL VOC format. This project expects the **VOC2007 and VOC2012 datasets** to be manually moved and organized as follows:

- `data/images/`: JPEG images from VOC2007 and VOC2012.
- `data/annotations/`: corresponding XML annotation files.

Ensure that the filenames of images and annotations match (e.g., `000001.jpg` and `000001.xml`).

**Note:** the test split of VOC2012 is not included due to missing annotations.

## Usage

### Training the Model

Run the main training script:

```bash
python miniyolo.py
```

This will start the training process using the dataset in `data/`, saving models and weights in the `trained/` directory.

### Model Conversion

After training, you can convert the trained model to TensorFlow Lite using:

```bash
python converter.py --path-weights PATH_WEIGHTS (.weights.h5)
or
python converter.py --path-model PATH_MODEL (.keras)
```


### Deployment

The converted model has been tested by uploading the TFLite model on [STM32 Edge AI Developer Cloud](https://stedgeai-dc.st.com/home) by STMicroelectronics on the STM32N6570-DK reaching 442 inferences/s, but with a low mAP.
