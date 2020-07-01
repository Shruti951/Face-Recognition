# first execute face_data.py and then execute
# python recognition.py --data data.pickle --image examples/mc.jpg 


import face_recognition
import argparse
import pickle
import cv2

#construct and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--data", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

#load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["data"], "rb").read())

#load the input image 
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the x and y coordinates of the face and encode
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)


names = []


for encoding in encodings:
	#match faces from the known dataset
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"

	#check whether you found your match
	if True in matches:
		
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		#loop over the matched indexes
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of
		#match counts
		name = max(counts, key=counts.get)
	
	# update the list of names
	names.append(name)

#loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

#show the output image
cv2.imshow("Image", image)
cv2.imwrite("./output/1.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
#cv2.waitKey(0)
