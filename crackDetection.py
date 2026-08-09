from ultralytics import YOLO
import cv2

model = YOLO("runs/classify/sdnet_crack_detection/yolov8_first_run/weights/best.pt")

test_image = "test_crack.jpg"

results = model(test_image)

for result in results:
    # Get highest confidence class index and confidence score
    top1_id = result.probs.top1  # Class index (0 or 1)
    top1_conf = result.probs.top1conf.item()  # Confidence (e.g. 0.982)
    class_name = result.names[top1_id]  # Class label ('cracked' or 'noncracked')

    print(f"File: {result.path}")
    print(f"Prediction: {class_name.upper()} ({top1_conf * 100:.2f}% confidence)\n")

    annotated_frame = result.plot()

    annotated_frame = cv2.resize(annotated_frame, (640,640))

    cv2.imshow("YOLOv8 Concrete Crack Classification", annotated_frame)
    cv2.waitKey(0)  # Press any key to close
    cv2.destroyAllWindows()