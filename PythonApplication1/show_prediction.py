import cv2
import os

# Folder where YOLO saved predictions
pred_dir = r"C:\Users\Khadiga Yahia\runs\detect\predict"

# List all predicted images
files = os.listdir(pred_dir)

# Choose any image (same idea as files[5])
image_path = os.path.join(pred_dir, files[0])

# Read image
img = cv2.imread(image_path)

# Show image
cv2.imshow("YOLO Prediction", img)

# Wait until key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
