from ultralytics import YOLO

class CrackAssessment:
    def __init__(self):
        """
        Responsible for initializing the model once to avoid
        reloading weights every time an image is evaluated.
        """

        model_path = "runs/classify/sdnet_crack_detection/yolov8_kg50_2ndrun(CLAHE)/weights/best.pt"

        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def assessImage(self, image):
        """
        Assesses the crack image and returns the crack assessment and confidence percent.

        :param image: Image Path
        :return dict: Crack Assessment and other metrics (confidence percent, etc)
        """

        results = self.model(image, verbose=False) # Verbose prevents Ultralytics console logs
        result = results[0]

        top1_id = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf.item())
        class_name = self.class_names[top1_id]
        raw_probs = {
            self.class_names[i]: float(prob)
            for i, prob in enumerate(result.probs.data.tolist())
        }

        return{
            "assessment": class_name.upper(),
            "class_id": top1_id,
            "confidence": top1_conf,
            "confidence_percent": round(top1_conf * 100, 2),
            "raw_probabilities": raw_probs
        }


