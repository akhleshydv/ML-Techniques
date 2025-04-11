import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set up dataset path
dataset_path = "./PetImages"

# Initialize counters
class_counts = {'cat': 0, 'dog': 0}
corrupted_images = 0
image_dimensions = []

# Analyze dataset
for class_dir in ['cat', 'dog']:
    dir_path = os.path.join(dataset_path, class_dir)
    for img_path in Path(dir_path).glob('*'):
        try:
            with Image.open(img_path) as img:
                class_counts[class_dir] += 1
                image_dimensions.append(img.size)
        except:
            corrupted_images += 1

# Print dataset summary
print(f"Class counts: {class_counts}")
print(f"Corrupted images: {corrupted_images}")

# Plot class distribution
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.show()

# Plot image dimensions
widths, heights = zip(*image_dimensions)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(widths, bins=20, color='blue', alpha=0.7)
plt.title('Image Width Distribution')
plt.xlabel('Width')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(heights, bins=20, color='green', alpha=0.7)
plt.title('Image Height Distribution')
plt.xlabel('Height')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Display sample images
def display_sample_images(class_name, num_samples=5):
    dir_path = os.path.join(dataset_path, class_name)
    sample_images = list(Path(dir_path).glob('*'))[:num_samples]
    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(sample_images):
        img = Image.open(img_path)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.show()

display_sample_images('cat')
display_sample_images('dog')