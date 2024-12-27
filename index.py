from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2 as cv


#  getting the  webcam
videoFeed = cv.VideoCapture(0)
# the segmentation object
segmentation = SelfiSegmentation()
while True:
    it_there,frame = videoFeed.read()
    frame = cv.flip(frame,1)
    #   converting to the rgb 
    convertedImage = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    removedBG = segmentation.removeBG(convertedImage,(240,0,250),0.8)
    #  stacking the orignal image and the removed background image
    stacked= cv.hconcat([frame,removedBG])
    #  showing the thing
    cv.imshow("Webcam",stacked)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
videoFeed.release()