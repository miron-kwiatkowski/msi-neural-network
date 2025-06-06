import os
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import jax.tree_util

# Define your model architecture (same as training)
class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=14)(x)
        return x

# Load model parameters
def load_params(path='model_params.npz'):
    with np.load(path, allow_pickle=True) as f:
        params = {k: f[k] for k in f.files}
    # Convert numpy arrays to JAX arrays
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    return params

# Load your label map (you can save and load from json or define manually)
label_map = {
    0: "Monster1",
    1: "Monster2",
    2: "Monster3",
    3: "Monster4",
    4: "Monster5",
    5: "Monster6",
    6: "Monster7",
    7: "Monster8",
    8: "Monster9",
    9: "Monster10",
    10: "Monster11",
    11: "Monster12",
    12: "Monster13",
    13: "Monster14"
}

app = Flask(__name__)
model = SimpleCNN()
params = load_params()

def preprocess_image(file_storage, target_size=(128,128)):
    # Read image from uploaded file
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return jnp.array(img)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        img = preprocess_image(file)
        if img is None:
            prediction = "Invalid image"
        else:
            logits = model.apply(params, img)
            pred_class = int(jnp.argmax(logits, axis=-1)[0])
            prediction = label_map.get(pred_class, f"Class {pred_class}")
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
