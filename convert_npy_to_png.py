import numpy as np
import os
from PIL import Image

# Pick your categories (must match the npy files you downloaded)
categories = ["alarm clock", "apple", "axe", "backpack", "baseball", "basket", "bat", "book", "butterfly", "cactus", "cake", "candle", "car", "cat", "crown", "cup", "diamond", "donut", "drums", "duck", "eraser", "eye", "eyeglasses", "fan", "fish", "flower", "grapes", "hammer", "hat", "hexagon", "hourglass", "house", "ice cream", "leaf", "lollipop", "moon", "mountain", "necklace", "pants", "pencil", "pizza", "rainbow", "scissors", "see saw", "snowman", "star", "sun", "table", "teapot", "tree", "umbrella", "wheel"]

for category in categories:
    print(f"Converting: {category}")
    data = np.load(f"dataset/{category}/{category}.npy")  # load npy file
    save_dir = f"dataset/{category}/"
    os.makedirs(save_dir, exist_ok=True)

    # Save first 500 images for each category
    for i, arr in enumerate(data[:500]):
        img = arr.reshape(28, 28)
        img = Image.fromarray(img.astype('uint8'))
        img = img.convert("RGB")  # convert grayscale to RGB
        img.save(f"{save_dir}/{category}_{i}.png")
