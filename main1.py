import tkinter as tk
import keras
import numpy as np
from PIL import ImageGrab, Image
import time
import os

# -------- Load the Teachable Machine SavedModel --------
model_path = "C:/Users/lenovo/Desktop/drawnguess/models/doodle_model.h5"  
model = keras.models.load_model(model_path)

# -------- Load Labels --------
labels_path = "models/labels.txt"
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        labels = [line.strip().split(" ", 1)[-1] for line in f]
else:
    print("⚠ labels.txt not found! Using index numbers instead.")
    labels = None

# -------- Prediction Function --------
def predict_image():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    img = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("RGB").resize((224,224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    # output = model(arr)
    # predictions = list(output.values())[0].numpy()[0]
    predictions = model.predict(arr, verbose=0)   # returns array of probabilities
    predictions = predictions[0]
    class_idx = int(np.argmax(predictions))
    confidence = float(predictions[class_idx])

    if labels:
        result_label.config(text=f"AI guesses: {labels[class_idx]} ({confidence*100:.1f}%)")
    else:
        result_label.config(text=f"AI guesses: {class_idx} ({confidence*100:.1f}%)")

def live_predict():
    try:
        canvas.postscript(file="drawing.eps")
        img = Image.open("drawing.eps").convert("L")
        img = img.resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        predictions = model.predict(img, verbose=0)[0]
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])
        #label = np.argmax(prediction)
        '''result_label.config(text=f"AI Guess: {label[class_idx]} ({confidence*100:.1f}%)")
    except:
        pass
    
    root.after(1000, live_predict)'''  # call every 1 second
        if labels:
            result_label.config(
                text=f"AI Guess: {labels[class_idx]} ({confidence*100:.1f}%)"
            )
        else:
            result_label.config(
                text=f"AI Guess: {class_idx} ({confidence*100:.1f}%)"
            )
        
    except Exception as e:
        print("Prediction error:", e)

    root.after(1000, live_predict)  # call every 1 second


# -------- Timer Function --------
time_left = 30
def countdown():
    global time_left
    if time_left > 0:
        time_left -= 1
        timer_label.config(text=f"Time left: {time_left}s")
        root.after(1000, countdown)
    else:
        predict_image()  # Auto-guess when time runs out


# -------- Tkinter UI --------
root = tk.Tk()
root.title("DrawNGuess - Round 1")

# -------- Drawing Functions --------
current_color = "black"
brush_size = tk.IntVar(value=3)

def paint(event):
    x1, y1 = (event.x - brush_size.get()), (event.y - brush_size.get())
    x2, y2 = (event.x + brush_size.get()), (event.y + brush_size.get())
    canvas.create_oval(x1, y1, x2, y2, fill=current_color, outline=current_color)

def set_color(new_color):
    global current_color
    current_color = new_color

def use_eraser():
    set_color("white")

def clear_canvas():
    canvas.delete("all")

# Canvas
canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.pack(pady=10)
canvas.bind("<B1-Motion>", paint)

# Brush size slider
size_slider = tk.Scale(root, from_=1, to=10, orient="horizontal", label="Brush Size", variable=brush_size)
size_slider.pack()

# Color buttons
color_frame = tk.Frame(root)
color_frame.pack()
for col in ["black", "red", "blue", "green", "orange", "purple"]:
    tk.Button(color_frame, text=col, bg=col, width=8, command=lambda c=col: set_color(c)).pack(side="left", padx=2)

# Eraser + Clear buttons
tools_frame = tk.Frame(root)
tools_frame.pack(pady=5)
tk.Button(tools_frame, text="Eraser", command=use_eraser).pack(side="left", padx=5)
tk.Button(tools_frame, text="Clear", command=clear_canvas).pack(side="left", padx=5)

# Submit button
submit_btn = tk.Button(root, text="Submit Drawing", command=predict_image)
submit_btn.pack(pady=5)

# Labels for results and timer
timer_label = tk.Label(root, text="Time left: 60s", font=("Arial", 14))
timer_label.pack()
result_label = tk.Label(root, text="AI will guess here", font=("Arial", 16))
result_label.pack(pady=10)

# Start countdown
countdown()

root.mainloop()

