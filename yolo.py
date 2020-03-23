import numpy as np
import cv2
import os

labelsPath = './yolo-coco/obj.names'
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = './yolo-coco/yolov3_obj_last.weights'
configPath = './yolo-coco/yolov3_obj.cfg'

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread('1.jpg')
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)
boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		if confidence >0.1:
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1,
	0.1)

if len(idxs) > 0:
	for i in idxs.flatten():
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (255, 0, 0), 2)
		image[y:y+h,x:x+w] = cv2.blur(image[y:y+h,x:x+w], (100, 100))

cv2.imwrite("Image.jpg", image)