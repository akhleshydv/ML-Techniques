# Step 1: Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from PIL import Image

# Step 2: Load Dataset
cat_images_path = './PetImages/Cat/'
dog_images_path = './PetImages/Dog/'
# Assuming the dataset is structured with two folders: Cat and Dog

cat_images = [os.path.join(cat_images_path, f) for f in os.listdir(cat_images_path)]
dog_images = [os.path.join(dog_images_path, f) for f in os.listdir(dog_images_path)]

data = pd.DataFrame({
    'image': cat_images + dog_images,
    'label': [0] * len(cat_images) + [1] * len(dog_images)  # 0 for cats, 1 for dogs
})

# Step 3: Display First Few Rows of Data
print(data.head())

# Step 4: Visualize a Few Random Images
def display_random_images(image_paths, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        img = Image.open(image_paths[i])
        ax.imshow(img)
        ax.set_title('Cat' if labels[i] == 0 else 'Dog')
        ax.axis('off')
    plt.show()

display_random_images(data['image'].values[:5], data['label'].values[:5])

# Step 5: Image Size & Aspect Ratio Distribution
def get_image_size(image_paths):
    sizes = []
    for path in image_paths:
        img = Image.open(path)
        sizes.append(img.size)  # (width, height)
    return sizes

sizes = get_image_size(data['image'].values)
sizes_df = pd.DataFrame(sizes, columns=['width', 'height'])
print(sizes_df.describe())

# Step 6: Class Distribution
sns.countplot(x='label', data=data)
plt.title('Class Distribution: Cats vs Dogs')
plt.xticks([0, 1], ['Cats', 'Dogs'])
plt.show()

# Step 7: Visualize Image Preprocessing (Resize and Normalize)
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    return img_array

preprocessed_img = preprocess_image(data['image'].values[0])
plt.imshow(preprocessed_img)
plt.title('Preprocessed Image (Normalized and Resized)')
plt.axis('off')
plt.show()

# Step 8: Data Augmentation Example
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

img = preprocess_image(data['image'].values[0])
img = np.expand_dims(img, axis=0)

augmented_images = datagen.flow(img)

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    aug_img = next(augmented_images)
    axes[i].imshow(aug_img[0])
    axes[i].axis('off')
plt.show()

# Step 9: Check for Corrupted Images
def check_corrupted_images(image_paths):
    corrupted = []
    for path in image_paths:
        try:
            img = Image.open(path)
            img.verify()  # Verify if the image can be opened
        except:
            corrupted.append(path)
    return corrupted

corrupted_images = check_corrupted_images(data['image'].values)
print(f"Corrupted Images: {corrupted_images}")

# Step 10: Prepare Data for Modeling (Split and Normalize)
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

X_train = np.array([preprocess_image(img_path) for img_path in train_df['image']])
X_val = np.array([preprocess_image(img_path) for img_path in val_df['image']])

y_train = np.array(train_df['label'])
y_val = np.array(val_df['label'])

# Step 11: Summary of Findings
# Class distribution: Check if there is an imbalance (if the dataset is mostly cats or dogs).
# Image size: Most images may require resizing to a standard format.
# Preprocessing: Image normalization and resizing is needed before feeding data into a model.
# Data Augmentation: Can be applied to avoid overfitting during model training.

print("EDA Complete. Data is ready for further model building and training.")
