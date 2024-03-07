import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Define the paths to the TensorFlow Lite model and label map
TF_LITE_MODEL = '4md.tflite'
LABEL_MAP = 'coco.names'
THRESHOLD = 0.5

# Load the TensorFlow Lite model and allocate tensors
interpreter = Interpreter(model_path=TF_LITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# Read the image file
image_path = 't2.jpg'  # Replace 't2.jpg' with your image file path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

# Perform inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the output tensors
boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
scores = interpreter.get_tensor(output_details[2]['index'])[0]

# Load label map
with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Annotate the image with detected objects
for score, box, class_ in zip(scores, boxes, classes):
    if score < THRESHOLD:
        continue

    class_name = labels[int(class_)]
    ymin, xmin, ymax, xmax = box
    xmin = int(xmin * image.shape[1])
    xmax = int(xmax * image.shape[1])
    ymin = int(ymin * image.shape[0])
    ymax = int(ymax * image.shape[0])

    color = (255, 0, 0)  # BGR color format
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(image, f'{class_name}: {score:.2f}', (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the annotated image
cv2.imshow('Object Detection Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
