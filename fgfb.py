import cv2
import json
import tensorflow 
import numpy as np
from keras.models import model_from_json
    
# Load the model architecture from the JSON file
with open('model.json', 'r') as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)

# Load the model weights
model.load_weights('weights.bin')

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess the image (resize, normalize, etc.) as needed by your model
        # For example, if your model expects a 224x224 RGB image:
        resized_frame = cv2.resize(frame, (224, 224))
        input_image = (resized_frame / 255.0)[np.newaxis, :, :, :]

        # Make predictions
        predictions = model.predict(input_image)

        # Find the index with the highest probability
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[0][predicted_class_index]

        # Print the predicted class and confidence
        print("Predicted Class Index:", predicted_class_index)
        print("Confidence:", confidence)

        # Display the captured frame
        cv2.imshow('Camera Feed', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    # Release the camera in case the user interrupts the script
    cap.release()
    cv2.destroyAllWindows()
