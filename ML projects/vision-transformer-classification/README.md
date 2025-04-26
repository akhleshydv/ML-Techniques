# Vision Transformer for Binary Image Classification

This project implements a Vision Transformer (ViT) model for binary image classification on a dataset of cat and dog images. The model is trained to distinguish between the two classes using deep learning techniques.

## Project Structure

```
vision-transformer-classification
├── src
│   ├── data
│   │   ├── dataset_preparation.py  # Functions for dataset preparation
│   │   └── transforms.py            # Image transformation functions
│   ├── models
│   │   └── vit_model.py              # Vision Transformer model definition
│   ├── training
│   │   ├── train.py                  # Training loop for the model
│   │   └── metrics.py                # Functions for calculating metrics
│   ├── utils
│   │   └── plot_curves.py            # Functions to plot loss and accuracy curves
│   └── main.py                       # Entry point for the application
├── data
│   └── PetImages                     # Directory containing cat and dog images
│       ├── cat
│       └── dog
├── requirements.txt                  # Required Python packages
├── README.md                         # Project documentation
└── .gitignore                        # Files to ignore in version control
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd vision-transformer-classification
   ```

2. **Install the required packages:**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   Place your cat and dog images in the `data/PetImages` directory, ensuring that the images are organized into `cat` and `dog` subdirectories.

## Usage

To train the Vision Transformer model, run the following command:

```bash
python src/main.py
```

This will initiate the training process, and the model will be trained on the provided dataset. During training, loss and accuracy curves will be plotted to visualize the model's performance.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- This project utilizes the Vision Transformer architecture as described in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.
- Special thanks to the contributors and the open-source community for their valuable resources and libraries.