
# python face_data.py --dataset dataset --data data.pickle


from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

#construct and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--data", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

#input to the dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))


knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	#extracting name of the person in the image from the dataset
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# select x and y coordinates of the face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	#facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		#add encoding and name on the dataset
		knownEncodings.append(encoding)
		knownNames.append(name)

#save the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["data"], "wb")
f.write(pickle.dumps(data))
f.close()
