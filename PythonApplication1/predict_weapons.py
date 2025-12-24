from ultralytics import YOLO
import cv2
import os

# ===============================
# 🔹 Load your trained YOLO model
# ===============================
model = YOLO(r"C:\Users\Khadiga Yahia\runs\detect\train8\weights\best.pt")

# ===============================
# 🔹 Folder containing validation images
# ===============================
val_folder = r"C:\Users\Khadiga Yahia\.kaggle\PythonApplication1\cctv_weapon_data\Dataset\images\val"

# ===============================
# 🔹 Loop over all images in the folder
# ===============================
for filename in sorted(os.listdir(val_folder)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(val_folder, filename)
        
        # Run YOLO prediction on the image
        results = model.predict(source=image_path, conf=0.10)
        
        # Only process images with detections
        if len(results[0].boxes) > 0:
            # Draw bounding boxes on the image
            result_img = results[0].plot()

            # Count detections per class
            classes_detected = results[0].names
            counts = {}
            for cls_id in results[0].boxes.cls:
                cls_name = classes_detected[int(cls_id)]
                counts[cls_name] = counts.get(cls_name, 0) + 1
            
            # Prepare text to display
            text_lines = [f"{k}: {v}" for k, v in counts.items()]
            y0, dy = 30, 30
            for i, line in enumerate(text_lines):
                y = y0 + i*dy
                cv2.putText(result_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Show image in a window
            cv2.imshow("Weapon Detection", result_img)
            
            # Display each image for 1 second (1000 ms)
            key = cv2.waitKey(1000)  # 1000 ms = 1 second
            if key == 27:  # Press ESC to exit early
                break

# Close all OpenCV windows
cv2.destroyAllWindows()
