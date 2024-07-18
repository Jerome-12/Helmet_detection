import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Helmet Detection")

CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

config_path = "D:/Data_Excel/AI/dataset/yolov3-helmet.cfg"
weights = "D:/Data_Excel/AI/dataset/yolov3-helmet.weights"
labels = open("D:/Data_Excel/AI/dataset/helmet.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights)

def model_output(image):
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    ln = [net.getLayerNames()[int(layer) - 1] for layer in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(ln)

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def detection_recognition(image):
    boxes, confidences, class_ids = model_output(image)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    font_scale = 1
    thickness= 1

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            cv2.rectangle(image, (x,y), (x+w, y+h), color=(240, 236, 10), thickness=thickness)
            text = f"{labels[class_ids[i]]}:{confidences[i]:.2f}"
            cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.35,color=(240, 236, 10), thickness=thickness)
           

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    st.pyplot(plt.gcf())

if __name__ == "__main__":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png","jfif"])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = np.asarray(bytearray(image_bytes), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if st.button("Submit"):
            detection_recognition(image)