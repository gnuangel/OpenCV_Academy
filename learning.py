import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Load YOLO weights and config files
classes = []  # Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load input image
image = cv2.imread("example.jpg")
height, width, channels = image.shape

# Prepare input image for YOLO network
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Perform forward pass to get output from YOLO
outs = net.forward(output_layers)

# Process the detections
for out in outs:
    for detection in out:
        scores = detection[5:]  # Class scores
        class_id = np.argmax(scores)  # Get the index of the class with the highest score
        confidence = scores[class_id]  # Confidence score
        if confidence > 0.5:  # If confidence is high enough
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw bounding box and label
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
