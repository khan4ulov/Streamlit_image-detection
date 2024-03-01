import cv2
import numpy as np
import streamlit as st
from PIL import Image

def post_process(frame, outs, img, classes, indexes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    boxes = []
    confidences = []
    classIDs = []
    detected_classes = set()
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(centerX - width / 2)
                top = int(centerY - height / 2)
                classIDs.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                detected_classes.add(classes[class_id])

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # Black color
    for i in indexes:
        x, y, w, h = boxes[i[0]]
        label = str(classes[classIDs[i[0]]])
        confi = str(round(confidences[i[0]], 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if label in detected_classes:
            cv2.putText(img, label + " " + confi, (x, y - 5), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)  # Highlight detected classes in red
        else:
            cv2.putText(img, label + " " + confi, (x, y - 5), font, 1, color, 2, cv2.LINE_AA)  # Set text color to black
    return img, detected_classes

def yolo_out(image):
    modelConf = "yolov3-tiny.cfg"
    modelWeights = "yolov3-tiny.weights"
    classesFile = "coco.names"

    net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    frame = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), 1, crop=False)
    net.setInput(blob)
    yolo_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(yolo_layers)

    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                centerX = int(detection[0] * frame.shape[1])
                centerY = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(centerX - width / 2)
                top = int(centerY - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))

    # Perform Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if indexes is empty
    if len(indexes) > 0:
        # Convert indexes to list of lists
        indexes = [[i] for i in indexes.flatten()]
    else:
        # Handle the case where no objects are detected
        return img, "", False

    # Post-process and draw bounding boxes
    result_img, detected_classes = post_process(frame, outs, img, classes, indexes)

    return result_img, detected_classes, True

def main():
    st.markdown("<h1 style='text-align: center; color: Black;'>YOLO Object Detection with Streamlit</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            result_img, detected_classes, objects_detected = yolo_out(uploaded_file)
            if objects_detected:
                st.image(result_img, caption='Processed Image', use_column_width=True)
                st.markdown(f"<p style='color: blue; font-size: 20px; font-weight: bold; font-style: italic;'>Detected Classes: {', '.join(detected_classes)}</p>", unsafe_allow_html=True)
            else:
                st.warning("No objects detected in the image.")
                original_img = np.array(Image.open(uploaded_file))
                st.image(original_img, caption='Original Image', use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()