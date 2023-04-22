import cv2

#Our Video
video = cv2.VideoCapture('Dashcam Pedestrian.mp4')

#Our pre-trained car classifier and pedestrian
classifier_file_car = 'car_detector.xml'
classifier_file_pedestrian = 'haarcascade_fullbody.xml'

#Create car classifier and pedestrian
car_tracker = cv2.CascadeClassifier(classifier_file_car)
pedestrian_tracker = cv2.CascadeClassifier(classifier_file_pedestrian)

#Runs forever until video ends
while True:
    read_successful, frame = video.read()
    if read_successful:
       grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect any size and type of cars or pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)#, scaleFactor1.1, minNeighbors=2)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around the cars and pedestrians
    for (x, y, w, h) in cars:
         cv2.rectangle(frame, (x+1, y+1), (x+w, y+h), (0, 0, 225), 3)
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 3)

    for (x, y, w, h) in pedestrian:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #Display box with cars and pedestrians when spotted
    cv2.imshow('Self Driving Car', frame)

    #To display longer
    key = cv2.waitKey(1)

    #Hit Q to stop video
    if key==81 or key==113:
        break
#Realse the VideoCapture

video.release()