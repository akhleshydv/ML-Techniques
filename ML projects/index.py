import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()  # Verify it's an image
        return True
    except:
        return False

def create_transforms(image_size=224):
    """
    Create a transformation pipeline for image preprocessing
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

def prepare_datasets():
    # Set up parameters
    image_size = 224
    dataset_path = "./PetImages"
    
    # Remove corrupted images first
    for class_dir in ['cat', 'dog']:
        dir_path = os.path.join(dataset_path, class_dir)
        for img_path in Path(dir_path).glob('*'):
            if not is_valid_image(img_path):
                print(f"Removing corrupted image: {img_path}")
                os.remove(img_path)
    
    # Create transform pipeline
    transform = create_transforms(image_size)
    
    # Load the full dataset
    try:
        full_dataset = ImageFolder(root=dataset_path, transform=transform)
        
        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Print dataset information
        print("\nDataset Summary:")
        print(f"Total images: {len(full_dataset)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"\nClasses: {full_dataset.classes}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"Error preparing datasets: {str(e)}")
        return None, None

if __name__ == "__main__":
    train_dataset, val_dataset = prepare_datasets()