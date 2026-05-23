import tkinter as tk
import tensorflow as tf
import keras
import numpy as np
from PIL import Image, ImageGrab
import time
import os

# Load the Teachable Machine model
model = keras.layers.TFSMLayer("best_doodle_model.h5", call_endpoint="serving_default")

# Load labels
with open("models/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Labels file from Teachable Machine
with open("models/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)  # batch dimension


# Run inference
    output = model(arr)  # returns a dict
    predictions = list(output.values())[0].numpy()[0]

    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]
    return labels[class_idx], confidence
















'''def predict_image(img_path):
    """Takes an image path, returns predicted class label"""
    img = Image.open(img_path).convert("RGB").resize((224,224))  
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 224, 224, 3)   # TM expects this format

    prediction = model.predict(arr)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    return labels[class_idx], confidence


def save_canvas():
    # Save the drawing area as an image
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    filename = f"player_drawing_{int(time.time())}.png"
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
    print("Saved:", filename)

    # Get AI prediction
    label, conf = predict_image(filename)
    result_label.config(text=f"AI guesses: {label} ({conf*100:.1f}%)")

# Tkinter window
root = tk.Tk()
root.title("DrawNGuess - Round 1")

canvas = tk.Canvas(root, width=300, height=300, bg="white")
canvas.pack()

# Drawing logic
def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black")

canvas.bind("<B1-Motion>", paint)

# Submit button
submit_btn = tk.Button(root, text="Submit Drawing", command=save_canvas)
submit_btn.pack(pady=10)

# Result label
result_label = tk.Label(root, text="AI will guess here", font=("Arial", 16))
result_label.pack()

root.mainloop()'''








'''def main():
    # Create the main window
    root = tk.Tk()
    root.title("DrawNGuess")
    root.geometry("800x600")   # window size
    root.config(bg="#f8f8f8")  # light gray background
    
    # Game title
    title = tk.Label(root, text="🎨 DrawNGuess 🎮", 
                     font=("Arial", 32, "bold"), 
                     bg="#f8f8f8", fg="#333")
    title.pack(pady=50)

    # Run window
    root.mainloop()

if __name__ == "__main__":
    main()'''