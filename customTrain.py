from ultralytics import YOLO


def customTrain():
    model = YOLO('yolov8m-cls.pt')

    results = model.train(
        data="/kaggle/working/sdnet_yolo_cls",
        epochs=100,
        imgsz=256,
        batch=32,
        workers=2,
        device=0,
        cache=True,
        project="/kaggle/working/sdnet_yolo_train/runs",
        name="yolov8_kg50_nc_3rdrun",

        # Augmentations
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.5,
        degrees=15,
        translate=0.1,
        scale=0.3,
        fliplr=0.7,

        # Hyper-Parameter Tuning
        patience=20

    )


if __name__ == '__main__':
    customTrain()