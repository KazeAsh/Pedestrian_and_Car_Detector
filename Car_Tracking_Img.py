import cv2

#Our Image
img_file = 'CarImg.jpg'

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'

#create opencv image
img = cv2.imread(img_file)

#Create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#convert to black and white
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect any size
cars = car_tracker.detectMultiScale(black_n_white)

#draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 225), 3)

#Display image
cv2.imshow('Clever Programmer Car Detector', img)

#To display
cv2.waitKey()