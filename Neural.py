import os

# Limit CPU threads
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4'

# Configure JAX memory and GPU
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import numpy as np
import cv2
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import random, device_put
import optax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define a simplified neural network
class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=14)(x)  # 14 classes for 14 monsters
        return x

# Load and preprocess data
def load_data(data_dir, target_size=(128, 128)):
    images, labels = [], []
    label_map = {}
    label_counter = 0
    for root, dirs, files in os.walk(data_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if folder not in label_map:
                label_map[folder] = label_counter
                label_counter += 1
            label = label_map[folder]
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                resize = cv2.resize(image, target_size)
                normalized = (resize / 255.0).astype(np.float32)
                images.append(normalized)
                labels.append(label)
                print(img_name + ' processed')
    return np.array(images), np.array(labels), label_map

# Training functions
def create_train_state(rng, learning_rate):
    model = SimpleCNN()
    params = model.init(rng, jnp.ones((1, 128, 128, 3)))  # Adjusted for input shape
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['images'])
        one_hot = jax.nn.one_hot(batch['labels'], num_classes=14)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, one_hot))
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch['images'])
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch['labels'])
    return accuracy

# Load data
data_dir = 'E:/Gothic'
images, labels, label_map = load_data(data_dir)

# Split into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Move data to GPU
train_images = device_put(train_images)
train_labels = device_put(train_labels)
test_images = device_put(test_images)
test_labels = device_put(test_labels)

# Initialize model and training state
rng = random.PRNGKey(0)
state = create_train_state(rng, learning_rate=0.001)

batch_size = 64
num_epochs = 30

test_accuracies = []

# Training loop
for epoch in range(num_epochs):
    # Shuffle training data
    perm = np.random.permutation(len(train_images))
    train_images = train_images[perm]
    train_labels = train_labels[perm]

    # Train in batches
    for i in range(0, len(train_images), batch_size):
        batch_images = train_images[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]
        batch = {
            'images': batch_images,
            'labels': batch_labels
        }
        state = train_step(state, batch)

    # Evaluate on the test set
    total_acc = 0
    count = 0
    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        batch = {
            'images': batch_images,
            'labels': batch_labels
        }
        acc = eval_step(state, batch)
        total_acc += acc * len(batch_labels)
        count += len(batch_labels)
    epoch_acc = float(total_acc / count)
    test_accuracies.append(epoch_acc)
    print(f"Epoch {epoch + 1}, Test accuracy: {epoch_acc:.4f}")

# Save trained model parameters
def save_params(params, path='model_params.npz'):
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), params)
    np.savez(path, **params_np)
    print(f"Model parameters saved to {path}")

save_params(state.params)

# Generate and save accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', linestyle='-')
plt.title("Test Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.close()
print("Accuracy plot saved to accuracy_plot.png")
