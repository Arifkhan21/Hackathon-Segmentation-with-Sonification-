import numpy as np
import cv2
import argparse
import time
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-p", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(2)
# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)






vid = cv2.VideoCapture(args["input"])
writer = None
counter = 0
success = 1
cnt=0
data_dict = {}

while success:
	success,image = vid.read()
	if cnt%5 == 0:
		(H, W) = image.shape[:2]
		print(H)
		print(W)

		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()
 
		# print("[INFO] YOLO took {:.6f} seconds".format(end - start))
		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			for detection in output:
			
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
 
				if confidence > args["confidence"]:
			
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
 
			
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
 
			
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])
		if len(idxs) > 0:
			c = 0
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				# print(x,y)
				# print(w,h)
				color = [0,0,255]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
				cv2.putText(image, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				arr = [] 
				x = 360 - x
				y = -1*(y-640)
				arr.append(x)
				arr.append(y)
				arr.append(w)
				arr.append(h)

				ind = int(cnt/5)
				if data_dict.get(ind) is None:
					data_dict[ind] = {}

					data_dict[ind][c] = arr 

				else:
					data_dict[ind][c] = arr

				c = c+1
				



			 
		




# print(idxs)
		# area = -10
		# area_pair=[]
		# dim_pair=[]
		# if len(idxs) > 0:
		# 	for i in idxs.flatten():
		# 			(x, y) = (boxes[i][0], boxes[i][1])
		# 			(w, h) = (boxes[i][2], boxes[i][3])
		# 			if w*h > area:
		# 				area_pair = []
		# 				dim_pair = []
		# 				area_pair.append(w)
		# 				area_pair.append(h)
		# 				dim_pair.append(x)
		# 				dim_pair.append(y)
		# 				area = w*h
		# print(area_pair)
		# print(area) 

		# if len(idxs) > 0:
		# 	color = [0,0,255]
		# 	cv2.rectangle(image, (dim_pair[0],dim_pair[1]), (dim_pair[0] + area_pair[0], dim_pair[1] + area_pair[1]), color, 2)		



		if writer is None:
		# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,(image.shape[1], image.shape[0]), True)

		# some information on processing single frame
		
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			# print("[INFO] estimated total time to finish: {:.4f}".format(
			# 	elap * total))

	# write the output frame to disk
		writer.write(image)
	else:
		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,(image.shape[1], image.shape[0]), True)

		
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			
		writer.write(image)





	cnt = cnt+1	
	
print(data_dict)
writer.release()
vid.release()

