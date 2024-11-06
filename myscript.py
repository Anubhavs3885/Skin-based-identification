import os
import cv2
import numpy as np
import random
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import LambdaCallback
import pandas as pd
import shutil

input_folder = 'Hand'
output_folder = 'multispectral_dataset'

def simulate_hyperspectral_from_rgb(rgb_image, wavelengths, num_bands):
    h, w, _ = rgb_image.shape
    hyperspectral_image = np.zeros((h, w, num_bands))
    for i in range(num_bands):
        sigma = (wavelengths[i] - 500) / 100
        hyperspectral_image[:, :, i] = gaussian_filter(rgb_image[:, :, 1], sigma=sigma)
        hyperspectral_image[:, :, i] += 0.3 * gaussian_filter(rgb_image[:, :, 0], sigma=sigma)
        hyperspectral_image[:, :, i] += 0.3 * gaussian_filter(rgb_image[:, :, 2], sigma=sigma)
    return hyperspectral_image

df = pd.read_csv("HandInfo.csv")
filtered_df = df[df['aspectOfHand'].isin(['dorsal right', 'dorsal left'])]
top_ids = filtered_df['id'].value_counts().head(4).index.tolist()
limited_images = filtered_df[filtered_df['id'].isin(top_ids)]
limited_images = limited_images.groupby('id').head(10)

train_data, test_data = train_test_split(limited_images, test_size=0.2, random_state=42, stratify=limited_images['id'])

def process_and_save_images(data, output_dir, num_bands, wavelengths):
    os.makedirs(output_dir, exist_ok=True)
    count = {}
    for _, row in data.iterrows():
        user_id = row['id']
        img_filename = row['imageName']
        user_folder = os.path.join(output_dir, str(user_id))
        os.makedirs(user_folder, exist_ok=True)
        img_path = os.path.join(input_folder, img_filename)
        if img_filename.endswith(('.jpg', '.png')) and os.path.exists(img_path):
            rgb_image = cv2.imread(img_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            rgb_image = rgb_image / 255.0
            hyperspectral_image = simulate_hyperspectral_from_rgb(rgb_image, wavelengths, num_bands)
            img_count = count.get(user_id, 0)
            hyperspectral_path = os.path.join(user_folder, f'hyperspectral_image_{img_count}.npy')
            np.save(hyperspectral_path, hyperspectral_image)

            count[user_id] = img_count + 1
            print(f"Saved hyperspectral image from {img_filename} for user {user_id}")

def create_dataset(data_dir, subset='train'):
    data_path = os.path.join(data_dir, subset)
    images = []
    labels = []
    users = sorted(os.listdir(data_path))
    user_to_idx = {user: idx for idx, user in enumerate(users)}

    for user in users:
        user_path = os.path.join(data_path, user)
        image_files = [f for f in os.listdir(user_path) if f.endswith('.npy')]

        for img_file in image_files:
            img_path = os.path.join(user_path, img_file)
            image = np.load(img_path)
            images.append(image)
            labels.append(user_to_idx[user])

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, user_to_idx

def create_model(num_users, num_bands):
    inputs = Input(shape=(1200, 1600, num_bands))
    patch_processor = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        BatchNormalization()
    ])
    processed_patches = patch_processor(inputs)
    x = Dense(512, activation='relu')(processed_patches)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_users, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir, num_bands, batch_size=16, epochs=50):
    X_train, y_train, user_to_idx = create_dataset(data_dir, 'train')
    mean = np.mean(X_train, axis=(0, 1, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 1, 2, 3), keepdims=True)
    X_train = (X_train - mean) / (std + 1e-8)

    X_test, y_test, _ = create_dataset(data_dir, 'test')
    X_test = (X_test - mean) / (std + 1e-8)

    num_users = len(user_to_idx)
    model = create_model(num_users, num_bands)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    max_training_accuracy = [0.0]

    def update_max_accuracy(epoch, logs):
        if logs.get('accuracy', 0.0) > max_training_accuracy[0]:
            max_training_accuracy[0] = logs['accuracy']

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        verbose=1,
        callbacks=[LambdaCallback(on_epoch_end=update_max_accuracy)]
    )

    print(f"Max training accuracy for num_bands={num_bands}: {max_training_accuracy[0]:.4f}")
    return max_training_accuracy[0]

accuracies = []
for num_bands in range(4, 17):
    wavelengths = np.linspace(400, 700, num_bands)
    train_output_folder = os.path.join(output_folder, 'train')
    test_output_folder = os.path.join(output_folder, 'test')

    process_and_save_images(train_data, train_output_folder, num_bands, wavelengths)
    process_and_save_images(test_data, test_output_folder, num_bands, wavelengths)

    test_accuracy = train_model(output_folder, num_bands)
    accuracies.append((num_bands, test_accuracy))

    shutil.rmtree(train_output_folder)
    shutil.rmtree(test_output_folder)

print("Accuracies for different num_bands:")
for num_bands, acc in accuracies:
    print(f"num_bands={num_bands}: test_accuracy={acc:.4f}")

num_bands_values, max_training_accuracies = zip(*accuracies)

plt.figure(figsize=(10, 6))
plt.plot(num_bands_values, max_training_accuracies, marker='o')
plt.title('Maximum Accuracy vs Number of Bands')
plt.xlabel('Number of Bands')
plt.ylabel('Maximum Accuracy')
plt.xticks(num_bands_values)
plt.grid()
plt.show()