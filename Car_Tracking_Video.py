import cv2

#Our Video
video = cv2.VideoCapture('Dashcam Pedestrian.mp4')

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'

#Create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#Runs forever until video ends
while True:
        read_successful, frame = video.read()
        if read_successful:
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
              break

      #detect any size
cars = car_tracker.detectMultiScale(grayscaled_frame)

      #draw rectangles around the cars
for (x, y, w, h) in cars:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 3)
    
#Display box with cars and pedestrians when spotted
      cv2.imshow('Self Driving Car', frame)

    #To display longer
      key = cv2.waitKey(1)

    #Hit Q to stop video
      if key==81 or key==113:
        break
#Realse the VideoCapture
video.release()