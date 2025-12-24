from ultralytics import YOLO
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# ==============================
# Load YOLO model
# ==============================
model = YOLO(r"C:\Users\Khadiga Yahia\runs\detect\train9\weights\best.pt")

# ==============================
# Main Window
# ==============================
root = Tk()
root.title("Weapon Detection System")

BG_COLOR = "#0f172a"
CARD_COLOR = "#1e293b"
TEXT_COLOR = "#e5e7eb"
ACCENT = "#38bdf8"

root.configure(bg=BG_COLOR)
root.geometry("1350x750")

# ==============================
# Title
# ==============================
title = Label(root,
              text="🔫 Weapon Detection Using YOLO",
              font=("Segoe UI", 22, "bold"),
              bg=BG_COLOR,
              fg=ACCENT)
title.pack(pady=10)

subtitle = Label(root,
                 text="Original vs Detection Result",
                 font=("Segoe UI", 11),
                 bg=BG_COLOR,
                 fg=TEXT_COLOR)
subtitle.pack()

# ==============================
# Create scrolling area
# ==============================
main_frame = Frame(root, bg=BG_COLOR)
main_frame.pack(fill=BOTH, expand=1)

canvas = Canvas(main_frame, bg=BG_COLOR, highlightthickness=0)
canvas.pack(side=LEFT, fill=BOTH, expand=1)

scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

content_frame = Frame(canvas, bg=BG_COLOR)
canvas.create_window((0, 0), window=content_frame, anchor="nw")

# ==============================
# Panels (cards)
# ==============================
left_frame = Frame(content_frame, bg=CARD_COLOR, bd=2, relief="ridge")
left_frame.grid(row=0, column=0, padx=25, pady=10)

right_frame = Frame(content_frame, bg=CARD_COLOR, bd=2, relief="ridge")
right_frame.grid(row=0, column=1, padx=25, pady=10)

Label(left_frame, text="Original Image", font=("Segoe UI", 12, "bold"),
      bg=CARD_COLOR, fg=ACCENT).pack(pady=5)

Label(right_frame, text="Detection Result", font=("Segoe UI", 12, "bold"),
      bg=CARD_COLOR, fg="#4ade80").pack(pady=5)

left_panel = Label(left_frame, bg=CARD_COLOR)
left_panel.pack(padx=10, pady=10)

right_panel = Label(right_frame, bg=CARD_COLOR)
right_panel.pack(padx=10, pady=10)

# ==============================
# Image resize helper
# ==============================
def auto_resize(img, max_w=600, max_h=500):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))


# ==============================
# Open + Detect Function
# ==============================
def open_and_detect():

    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not path:
        return

    original = cv2.imread(path)

    # Resize safely
    original_resized = auto_resize(original)

    # Run YOLO
    results = model.predict(original, conf=0.25)
    detected = results[0].plot()

    detected_resized = auto_resize(detected)

    # ---- Show original ----
    orig_rgb = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
    orig_img = ImageTk.PhotoImage(Image.fromarray(orig_rgb))
    left_panel.configure(image=orig_img)
    left_panel.image = orig_img

    # ---- Show detection ----
    det_rgb = cv2.cvtColor(detected_resized, cv2.COLOR_BGR2RGB)
    det_img = ImageTk.PhotoImage(Image.fromarray(det_rgb))
    right_panel.configure(image=det_img)
    right_panel.image = det_img


# ==============================
# Button
# ==============================
btn = Button(root,
             text="📂 Choose Image",
             command=open_and_detect,
             font=("Segoe UI", 14, "bold"),
             bg="#0ea5e9",
             fg="white",
             relief="flat",
             width=18)
btn.pack(pady=10)

root.mainloop()
