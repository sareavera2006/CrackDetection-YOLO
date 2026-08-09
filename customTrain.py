from ultralytics import YOLO

def customTrain():
    # Sixth Run Parameters
    # model = yolov8n-cls (nano version)
    # imgz = 384 (increase the resolution to handle thinner cracks)
    # cache=True (helps speed up CPU training by caching images into ram
    # Additional spatial & rotation augmentations

    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data="./sdnet_yolo_cls",
        epochs=5,
        imgsz=384,
        batch=16,
        workers=2,
        device="cpu",
        cache=True,
        project="sdnet_crack_detection",
        name="yolov8_d3k_6thrun",

        # Augmentations (Aids in generalization)
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5,
        scale=0.2,
        hsv_v=0.4,
        hsv_s=0.3
    )

if __name__ == '__main__':
    customTrain()