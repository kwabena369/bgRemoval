import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Initialize video feed
videoFeed = cv.VideoCapture(0)

while True:
    ret, frame = videoFeed.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = segmentation.process(rgb_frame)

    # Create a mask and apply it to the frame
    condition = results.segmentation_mask > 0.5
    bg_color = (240, 0, 250)
    bg_image = np.zeros(frame.shape, dtype=np.uint8)
    bg_image[:] = bg_color
    output_image = np.where(condition[..., None], frame, bg_image)

    # Stack the original and output images
    stacked = cv.hconcat([frame, output_image])

    # Display the result
    cv.imshow("Webcam", stacked)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

videoFeed.release()
cv.destroyAllWindows()