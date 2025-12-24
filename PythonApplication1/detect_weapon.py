from ultralytics import YOLO
import cv2
import os

# ============================================================
# 1️⃣ Load trained YOLO model (Weapon Detection)
# ============================================================
model = YOLO(
    r"C:\Users\Khadiga Yahia\runs\detect\train8\weights\best.pt"
)

# ============================================================
# 2️⃣ Source folder (validation images)
# ============================================================
source_folder = (
    r"C:\Users\Khadiga Yahia\.kaggle\PythonApplication1"
    r"\cctv_weapon_data\Dataset\images\val"
)

# ============================================================
# 3️⃣ Output folder (predictions will be saved here)
# ============================================================
output_folder = (
    r"C:\Users\Khadiga Yahia\runs\detect\weapon_predict"
)

os.makedirs(output_folder, exist_ok=True)

# ============================================================
# 4️⃣ Loop through images and run prediction
# ============================================================
for filename in sorted(os.listdir(source_folder)):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(source_folder, filename)

        # Run YOLO prediction
        results = model.predict(
            source=image_path,
            imgsz=640,
            conf=0.25
        )

        # Plot detections on image
        result_image = results[0].plot()

        # Save predicted image
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, result_image)

        # Show image (optional – زي الفيديو)
        cv2.imshow("Weapon Detection", result_image)

        # Show each image for 1 second
        if cv2.waitKey(1000) == 27:  # ESC to exit
            break

cv2.destroyAllWindows()

print("✅ Weapon detection finished successfully!")
