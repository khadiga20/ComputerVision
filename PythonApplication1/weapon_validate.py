from ultralytics import YOLO

# ============================================================
# 1️⃣ Load trained weapon detection model
# ============================================================
model = YOLO(
    r"C:\Users\Khadiga Yahia\runs\detect\train8\weights\best.pt"
)

# ============================================================
# 2️⃣ Run validation on the dataset
# (Equivalent to: !yolo val ...)
# ============================================================
metrics = model.val(
    data=r"C:\Users\Khadiga Yahia\.kaggle\PythonApplication1\cctv_weapon_data\Dataset\data.yaml",
    imgsz=640
)

print("✅ Validation finished successfully!")
