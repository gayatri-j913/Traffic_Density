import cv2
import numpy as np
from ultralytics import YOLO

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('models/best.pt')

# Define the threshold for considering traffic as heavy
heavy_traffic_threshold = 10

# Define the vertices for the quadrilaterals
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Define the vertical range for the slice and lane threshold
x1, x2 = 325, 635 
lane_threshold = 609

# Define the positions for the text annotations on the image
text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)

# Define font, scale, and colors for the annotations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)    # White color for text
background_color = (0, 0, 255)  # Red background for text

# Read the image
image = cv2.imread(r'C:\Users\ankit\Downloads\YOLOv8_Traffic_Density_Estimation-master\YOLOv8_Traffic_Density_Estimation-master\Vehicle_Detection_Image_Dataset\sample_image.jpg')

if image is None:
    print("Error: Image not found or unable to read.")
else:
    # Create a copy of the original image to modify
    detection_frame = image.copy()

    # Black out the regions outside the specified vertical range
    detection_frame[:x1, :] = 0  # Black out from top to x1
    detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the image

    # Perform inference on the modified image
    results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
    processed_frame = results[0].plot(line_width=1)

    # Restore the original top and bottom parts of the image
    processed_frame[:x1, :] = image[:x1, :].copy()
    processed_frame[x2:, :] = image[x2:, :].copy()

    # Draw the quadrilaterals on the processed image
    cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

    # Save the processed image to a file
    output_path = 'processed_image_with_bounding_boxes.jpg'
    cv2.imwrite(output_path, processed_frame)
    print(f"Processed image saved as {output_path}")

    # Display the processed image
    cv2.imshow('Image with Bounding Boxes', processed_frame)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()
