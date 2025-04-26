import os
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from .transforms import create_transforms

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()  # Verify it's an image
        return True
    except:
        return False

def prepare_datasets(dataset_path="./data/PetImages", image_size=224):
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