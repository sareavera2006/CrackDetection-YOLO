from crackDetection import CrackAssessment
from crackMeasurement import CrackAnalysis
import cv2

"""
INTEGRATE GUI CODE IN THE MAIN 
"""

def main():

    # Initialization
    detector = CrackAssessment() # Initializes the model
    analyzer = CrackAnalysis() # Initializes the crack measurement module
    imagePath = "test3.jpg"

    result = detector.assessImage("test3.jpg")

    classLabel = result["assessment"]
    confidenceValue = result["confidence_percent"]

    print(classLabel)
    print(confidenceValue)

    if classLabel == "NONCRACKED":

        print("There is no crack within the image") # PLACEHOLDER. REPLACE WITH GUI CODE.

    elif classLabel == "CRACKED":

        print("There is a crack within the image")

        measurementResults = analyzer.analyzeCrackPixels(imagePath, threshold_method='otsu')

        print(f"Crack Area:     {measurementResults['crack_area_pixels']} px²")
        print(f"Crack Length:   {measurementResults['crack_length_pixels']} px")
        print(f"Maximum Width:  {measurementResults['max_width_pixels']} px")
        print(f"Average Width:  {measurementResults['avg_width_pixels']} px")

        # Save processed image showing crack region (blue) and centerline (red)
        cv2.imwrite("crack_analysis_overlay.jpg", measurementResults["visualization"])

if __name__ == "__main__":
    main()




