# Crack Detection & Size Measurement -- README File


---

## ⚡ Quick Start

Install the required dependencies before running any scripts:

```bash
!pip install ultralytics
```

---

## 📁 Dataset Setup

The SDNET2018 follows the following folder structure:
```text 
Dataset/
├── Decks/
│   ├── cracked/
│   └── non-cracked/
├── Pavements/
│   ├── cracked/
│   └── non-cracked/
├── Walls/
│   ├── cracked/
│   └── non-cracked/
```

However, the YOLO model would expect the following folder structure:
```text

# SDNET-YOLO Based Dataset 
Dataset/
├── train/
│   ├── cracked/
│   └── non-cracked/
├── val/
│   ├── cracked/
│   └── non-cracked/
├── test/
│   ├── cracked/
│   └── non-cracked/
```

So, datasetSetup.py is responsible for ensuring that the SDNET2018 is properly configured to meet the necessary folder structure of the YOLO model. 

```python
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

CLASS_SIZE = 3000
```

This is conducted by first establishing the ratio for each of the folder and the class size. The ratio is based on a 80:10:10 split with 80% of the dataset is allocated for training the model, while 10% is allocated for validating the model and the last 10% is for testing the model. 

One can change this parameter by tweaking the TRAIN_RATIO and VAL_RATIO variable. Additionally, the CLASS_SIZE can be changed to increase or decrease the number of samples. Its important that the number of samples in cracked folder and non-cracked folder is properly balanced. This ensures that the model is not bias.

***Example:***
```text
RATIO - 80:10:10
Class Size - 3000

Train : 2400
Val : 600
Test : 600
```
---

## 🏋️ Model Training 

---
<p style="color: yellow; font-style: italic;">In order to run the following code, ensure that ultralytics is properly installed. Kindly refer to the <b>Quick Setup</b> section.</p>

---

Once ultralytics is properly installed and the dataset is properly configured, one can run the customTrain.py file to begin model training. 

The contents will be further explained in the following, should any alterations be needed.

**Choice of Model**
```python
model = YOLO("yolov8n-cls.pt")
```
Currently, since there is no bounding box on the SDNET2018 dataset, the YOLO model to be used is the classification model. The different kinds of classification model can be configured in this line of code. Kindly refer to the ultralytics documentation on the YOLOv8 for more on that. 

**Model Training Variables**
```python
results = model.train(
        data="./sdnet_yolo_cls",
        epochs=50,
        imgsz=384,
        batch=16,
        workers=2,
        device="cpu",
        cache=True,
        project="sdnet_crack_detection",
        name="yolov8_run",

        # Augmentations (Aids in generalization)
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5,
        scale=0.2,
        hsv_v=0.4,
        hsv_s=0.3
    )
```
This line of code is responsible for training the loaded model. 

'data' - this is where the path to the dataset should be written. <br>
'epochs' - this refers to the number of iterations that the model would undergo training. <br>
imgsz - this simply refers to the resolution of which the model will use (256x256, 384x384, etc etc). <br>
'batch' - this refers to how many images are being processed at a time. In this case, 16 images are being processed by the model.<br>
'workers' - this refers to how many CPU threads is allocated for data loading and preprocessing. 
'device' - This is where it dictates whether the model training will run on either the CPU or GPU. ("cpu" - cpu, 0 - GPU). <br>
'cache' - Takes either true or false, dictates whether the data is to be loaded into the RAM or not. This helps in speeding up training, but does take up a bit of RAM as a result. <br>
'project' & 'run' - These two simply create a folder to store the runs in. Highly recommended to rename the 'name' variable to indicate what numbered run the training is on.<br>

**Augmentations** 

- hsv_h (Hue): Adjusts the color tone by a fraction of the color wheel (Default: 0.015). 
- hsv_s (Saturation): Alters color intensity (Default: 0.7). 
- hsv_v (Value/Brightness): Modifies image brightness (Default: 0.4). 
- bgr: Probability of flipping channels from RGB to BGR to handle incorrect channel ordering (Default: 0.0).
- degrees: Rotates the image randomly within a degree range (Default: 0.0). 
- translate: Shifts the image horizontally/vertically by a fraction of its size (Default: 0.1). 
- scale: Zooms in or out by a gain factor (Default: 0.5). 
- shear: Slants the image along an axis (Default: 0.0). 
- perspective: Applies a random 3D perspective distortion (Default: 0.0). 
- flipud: Probability of flipping the image upside-down (Default: 0.0). 
- fliplr: Probability of flipping the image left-to-right (Default: 0.5).

--- 
## Crack Detection 

Running the crack detection file requires training the model. After training the model, link the path of the model to the model variable. 

Following Example
```text

model_path = "runs/classify/sdnet_crack_detection/yolov8_kg50_2ndrun(CLAHE)/weights/best.pt"

```

After the path has been properly cited, attach the path of the desired image to the image_path variable. The model will then provide the assessment and the corresponding confidence value. 

---
## Crack Measurement

The crack measurement module is configured to accept the following: image path, the type of threshold method to be used, and the pixel per millimeter value. The image path hands the path of the image to the module, while the threshold method type that the module will use can be dictated to either apply adaptive gaussian or the otsu method. The threshold type is set to adaptive by default. The pixel per millimeter is also set to 0 by default. This means that the metrics returned for the crack's length and width will be measured in pixels. Currently no configuration is put in place for the proper conversion from pixel to millimeter. 

--- 
## Graphic User Interface

This can be loaded by running the **main.py** python file. 
<img width="1430" height="882" alt="image" src="https://github.com/user-attachments/assets/415be672-e0fe-4614-9e3b-5d39e295b7a5" />

In order to initiate the crack detection and measurement of the program, the user is prompted to select an image from their files.
<img width="1432" height="882" alt="image" src="https://github.com/user-attachments/assets/ccbc766a-8137-49b6-ab2b-b25ca33e548c" />

Click 'Run Analysis' to generate the crack assessment along with the confidence level. The assessment dictates whether a crack is present within an image and its confidence level. Additionally, this would also generate the measurement of the crack's length and width based on pixels. 
<img width="1600" height="882" alt="image" src="https://github.com/user-attachments/assets/414d8753-b0a9-4e29-8d9e-0d0f1fc9c976" />



