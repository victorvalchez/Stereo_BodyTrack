import numpy as np
import os
import cv2

filename1 = './media/video_cam0.mp4'
filename2 = './media/video_cam1.mp4'
frames_per_second = 30.0
res = '720p'

record = str(input('Record video? (y/n): ').lower())
if record == 'y':
    record = True
else:
    record = False

# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# Grab resolution dimensions and set video capture to it.
def get_dims(cap, res='720p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'H264'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

# Initialize video capture for both webcams
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Initialize video writers for both output files
out0 = cv2.VideoWriter(filename1, get_video_type(filename1), frames_per_second, get_dims(cap0, res))
out1 = cv2.VideoWriter(filename2, get_video_type(filename2), frames_per_second, get_dims(cap1, res))

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if ret0:
        if record:
            out0.write(frame0)
        cv2.imshow('Webcam 0', frame0)

    if ret1:
        if record:
            out1.write(frame1)
        cv2.imshow('Webcam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
out0.release()
out1.release()
cv2.destroyAllWindows()