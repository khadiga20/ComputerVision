from ultralytics import YOLO
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

# ==============================
# Custom Paths & Settings
# ==============================
MODEL_PATH = r"C:\Users\MEGA\Downloads\ComputerVision\runs\detect\train9\weights\best.pt"
model = YOLO(MODEL_PATH)

# ==============================
# Color Palette
# ==============================
BG = "#000000"
CARD = "#111111"
TEXT = "#f0f0f0"
ACCENT1 = "#3600b6"  
ACCENT2 = "#b40101"  
ACCENT3 = "#e9ff42"  
ACCENT4 = "#170072"
PANEL_WIDTH = 550
PANEL_HEIGHT = 350

# ==============================
# Main Window
# ==============================
root = Tk()
root.title("Advanced Weapon Detection System")
root.geometry("1200x700")
root.configure(bg=BG)

# ==============================
# Fonts
# ==============================
TITLE_FONT = ("SF Pro Display", 24, "bold")
SUBTITLE_FONT = ("SF Pro Display", 12, "bold")
BUTTON_FONT = ("SF Pro Display", 14, "bold")
CARD_TITLE_FONT = ("SF Pro Display", 14, "bold")
LABEL_FONT = ("SF Pro Display", 10)

# ==============================
# Animated Title
# ==============================
title_label = Label(root, text="🔫 ADVANCED WEAPON DETECTION",
                    font=TITLE_FONT, bg=BG, fg=ACCENT1)
title_label.pack(pady=5)

subtitle_label = Label(root, text="Detect weapons from Images or Videos in Real-Time",
                       font=SUBTITLE_FONT, bg=BG, fg=TEXT)
subtitle_label.pack(pady=(0, 10))

def animate_title(i=0):
    colors = [ACCENT1, ACCENT2, ACCENT3, ACCENT4]
    title_label.config(fg=colors[i % len(colors)])
    root.after(1500, animate_title, i+1)
animate_title()

# ==============================
# Panels (Before / After)
# ==============================
panel_frame = Frame(root, bg=BG)
panel_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)

def create_card(parent, title, color):
    card = Frame(parent, bg=CARD, bd=2, relief="ridge", width=PANEL_WIDTH, height=PANEL_HEIGHT+40)
    card.pack(side=LEFT, padx=5, pady=5)
    card.pack_propagate(False)  # Prevent auto resizing
    card_title = Label(card, text=title, font=CARD_TITLE_FONT, bg=CARD, fg=color)
    card_title.pack(pady=5)
    panel = Label(card, bg="#1a1f2b", bd=2, relief="sunken", width=PANEL_WIDTH, height=PANEL_HEIGHT)
    panel.pack()
    return panel

    return panel

left_panel = create_card(panel_frame, "📷 ORIGINAL", ACCENT3)
right_panel = create_card(panel_frame, "🎯 DETECTED", ACCENT2)

# ==============================
# Status Label
# ==============================
status_label = Label(root, text="Ready - Load an image or video", font=LABEL_FONT, bg=BG, fg=TEXT)
status_label.pack(pady=5)

# ==============================
# Resize Helper
# ==============================


def resize_to_panel(img, panel):
    h, w = img.shape[:2]
    scale = min(PANEL_WIDTH / w, PANEL_HEIGHT / h)
    new_w, new_h = int(w*scale), int(h*scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# ==============================
# IMAGE & VIDEO DETECTION
# ==============================
video_cap = None
video_running = False

def detect_image(path):
    global left_panel, right_panel
    status_label.config(text="🔄 Processing Image...", fg=ACCENT1)
    root.update_idletasks()
    try:
        img = cv2.imread(path)
        resized = resize_to_panel(img, left_panel)
        results = model.predict(img, conf=0.25, verbose=False)
        detected = results[0].plot()
        detected_resized = resize_to_panel(detected, right_panel)

        # Original
        orig_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        orig_img = ImageTk.PhotoImage(Image.fromarray(orig_rgb))
        left_panel.configure(image=orig_img)
        left_panel.image = orig_img

        # Detected
        det_rgb = cv2.cvtColor(detected_resized, cv2.COLOR_BGR2RGB)
        det_img = ImageTk.PhotoImage(Image.fromarray(det_rgb))
        right_panel.configure(image=det_img)
        right_panel.image = det_img

        status_label.config(text=f"✅ Image Detection Complete | Objects: {len(results[0].boxes)}", fg=ACCENT3)
    except Exception as e:
        status_label.config(text="❌ Detection Failed", fg=ACCENT2)
        print(e)

def detect_video(path):
    global video_cap, video_running
    video_cap = cv2.VideoCapture(path)
    video_running = True
    status_label.config(text="🔄 Processing Video...", fg=ACCENT4)

    while video_running:
        ret, frame = video_cap.read()
        if not ret:
            status_label.config(text="✅ Video Finished", fg=ACCENT4)
            video_cap.release()
            video_running = False
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        detected_frame = results[0].plot()
        detected_resized = resize_to_panel(detected_frame, right_panel)
        orig_resized = resize_to_panel(frame, left_panel)

        # Original frame
        orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
        orig_img = ImageTk.PhotoImage(Image.fromarray(orig_rgb))
        left_panel.configure(image=orig_img)
        left_panel.image = orig_img

        # Detected frame
        det_rgb = cv2.cvtColor(detected_resized, cv2.COLOR_BGR2RGB)
        det_img = ImageTk.PhotoImage(Image.fromarray(det_rgb))
        right_panel.configure(image=det_img)
        right_panel.image = det_img

        # Pause to allow GUI updates
        cv2.waitKey(30)

def detect_media():
    global video_running, video_cap
    path = filedialog.askopenfilename(
        filetypes=[("Image & Video Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv")]
    )
    if not path:
        return

    # Stop any previous video
    if video_running and video_cap is not None:
        video_running = False
        video_cap.release()

    if path.lower().endswith(('.jpg', '.jpeg', '.png')):
        detect_image(path)
    else:
        # Start video detection in a separate thread
        threading.Thread(target=detect_video, args=(path,), daemon=True).start()

# ==============================
# Buttons
# ==============================
button_frame = Frame(root, bg=BG)
button_frame.pack(pady=5)

def create_button(parent, text, color, command):
    btn = Button(parent, text=text, command=command, font=BUTTON_FONT,
                 bg=color, fg="white", relief="flat", width=25, cursor="hand2")
    btn.pack(side=LEFT, padx=5)
    return btn

detect_btn = create_button(button_frame, "📂 Upload & Detect", ACCENT1, detect_media)

root.mainloop()
