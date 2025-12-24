# ============================================================
# 🔍 Object Detection on Video (Weapons Detection)
# ============================================================

from ultralytics import YOLO
import cv2

# 1️⃣ Load your trained model
model = YOLO(r"C:\Users\Khadiga Yahia\runs\detect\train8\weights\best.pt")

# 2️⃣ Path to your video
video_path = r"C:\Users\Khadiga Yahia\.kaggle\PythonApplication1\cctv_weapon_data\Dataset\video\input_video.mp4"

# 3️⃣ Output video path
output_path = r"C:\Users\Khadiga Yahia\.kaggle\PythonApplication1\cctv_weapon_data\Dataset\video\output_video.mp4"

# 4️⃣ Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# 5️⃣ Define VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 6️⃣ Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    results = model.predict(frame, conf=0.25)

    # Draw boxes on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Weapons Detection", annotated_frame)

    # Write frame to output video
    out.write(annotated_frame)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7️⃣ Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
