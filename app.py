
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import json
from pathlib import Path

# =====================================
# LOAD MODEL AND CLASS NAMES
# =====================================

MODEL_PATH = Path("models/best_model.keras")
CLASS_PATH = Path("models/class_names.json")

print("Loading Omani Cuisine model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print("Number of classes:", len(class_names))


# =====================================
# APPLICATION
# =====================================

root = tk.Tk()
root.title("Omani Cuisine Recognition System")
root.geometry("850x700")
root.configure(bg="#F4F6F8")

title = tk.Label(
    root,
    text="OMANI CUISINE RECOGNITION",
    font=("Arial", 23, "bold"),
    bg="#F4F6F8",
    fg="#17324D"
)
title.pack(pady=20)

subtitle = tk.Label(
    root,
    text="AI-Based Food Image Classification Using MobileNetV2",
    font=("Arial", 11),
    bg="#F4F6F8",
    fg="#555555"
)
subtitle.pack()


# =====================================
# IMAGE DISPLAY
# =====================================

image_label = tk.Label(
    root,
    text="Upload an Omani food image",
    width=38,
    height=13,
    bg="white",
    relief="groove"
)
image_label.pack(pady=20)

result_label = tk.Label(
    root,
    text="Prediction results will appear here",
    font=("Arial", 14, "bold"),
    bg="#F4F6F8",
    fg="#17324D",
    justify="center"
)
result_label.pack(pady=15)

details_label = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    bg="#F4F6F8",
    fg="#333333",
    justify="left"
)
details_label.pack(pady=10)


# =====================================
# IMAGE PREDICTION
# =====================================

def upload_image():

    file_path = filedialog.askopenfilename(
        title="Select Food Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")
        ]
    )

    if not file_path:
        return

    try:

        # Display uploaded image
        display_image = Image.open(file_path).convert("RGB")
        display_image.thumbnail((320, 250))

        photo = ImageTk.PhotoImage(display_image)

        image_label.configure(
            image=photo,
            text=""
        )

        image_label.image = photo

        # Same preprocessing as evaluation script
        image = tf.keras.utils.load_img(
            file_path,
            target_size=(224, 224)
        )

        image = tf.keras.utils.img_to_array(image)

        image = np.expand_dims(
            image.astype(np.float32),
            axis=0
        )

        # Model prediction
        predictions = model.predict(
            image,
            verbose=0
        )[0]

        top_indices = np.argsort(predictions)[::-1][:3]

        best_index = int(top_indices[0])

        predicted_class = class_names[best_index]

        category, dish = predicted_class.split("/", 1)

        confidence = float(predictions[best_index]) * 100

        result_label.configure(
            text="PREDICTED DISH: " +
                 dish.replace("_", " ").title(),
            fg="#176B45"
        )

        details = (
            "Category: " + category.replace("_", " ").title()
            + "\nModel score: " + f"{confidence:.2f}%"
            + "\n\nTOP 3 PREDICTIONS\n"
        )

        for index in top_indices:

            details += (
                class_names[index].replace("/", " - ")
                .replace("_", " ").title()
                + " : "
                + f"{predictions[index] * 100:.2f}%"
                + "\n"
            )

        details += (
            "\nNote: Scores represent model outputs, "
            "not guaranteed real-world probabilities."
        )

        details_label.configure(text=details)

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# =====================================
# BUTTON
# =====================================

upload_button = tk.Button(
    root,
    text="UPLOAD FOOD IMAGE",
    command=upload_image,
    font=("Arial", 13, "bold"),
    bg="#17324D",
    fg="white",
    activebackground="#285577",
    activeforeground="white",
    padx=25,
    pady=12,
    cursor="hand2"
)

upload_button.pack(pady=15)

footer = tk.Label(
    root,
    text="Omani Cuisine Dataset | 50 Food Classes | MobileNetV2",
    font=("Arial", 9),
    bg="#F4F6F8",
    fg="#777777"
)

footer.pack(side="bottom", pady=15)

root.mainloop()
