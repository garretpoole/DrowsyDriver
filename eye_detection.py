import cv2 as cv
import dlib
import numpy as np
import sys
import time

if len(sys.argv) != 2:
    print('argument length=', len(sys.argv))
    print("usage: python3 [script name] [video file name or 0 for camera input]")
    exit(0)

file = sys.argv[1]
if file == '0':
    file = 0

cap = cv.VideoCapture(file)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cv.namedWindow('framer')
#init start time and total drowsy frames for video
start = 0
drowsyFrames = 0
totalFrames = 0
while True:

    ret, frame = cap.read()

    if not ret:
        print('read not successful')
        break
    
    totalFrames += 1
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        for i in range(0, 68):
            loc = (landmarks.part(i).x, landmarks.part(i).y)
            cv.circle(frame, loc, 1, (0, 255, 0), 1)
        right_eye_right_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye_left_point = (landmarks.part(39).x, landmarks.part(39).y)

        right_eye_39 = (landmarks.part(38).x, landmarks.part(38).y)
        right_eye_41 = (landmarks.part(40).x, landmarks.part(40).y)

        right_eye_38 = (landmarks.part(37).x, landmarks.part(37).y)
        right_eye_42 = (landmarks.part(41).x, landmarks.part(41).y)

        right_top_mid = ((right_eye_39[0] + right_eye_38[0])//2,
                         (right_eye_38[1] + right_eye_39[1])//2)
        right_bottom_mid = ((right_eye_41[0] + right_eye_42[0])//2,
                            (right_eye_41[1] + right_eye_42[1])//2)

        left_eye_right_point = (landmarks.part(42).x, landmarks.part(42).y)
        left_eye_left_point = (landmarks.part(45).x, landmarks.part(45).y)

        left_eye_44 = (landmarks.part(43).x, landmarks.part(43).y)
        left_eye_48 = (landmarks.part(47).x, landmarks.part(47).y)

        left_eye_45 = (landmarks.part(44).x, landmarks.part(44).y)
        left_eye_47 = (landmarks.part(46).x, landmarks.part(46).y)

        left_top_mid = ((left_eye_44[0] + left_eye_45[0])//2,
                        (left_eye_44[1] + left_eye_45[1])//2)
        left_bottom_mid = ((left_eye_47[0] + left_eye_48[0])//2,
                           (left_eye_47[1] + left_eye_48[1])//2)

        horizontal_dist_right = right_eye_left_point[0] - \
                right_eye_right_point[0]
        right_vert_dist_1 = right_eye_42[1] - right_eye_38[1]
        right_vert_dist_2 = right_eye_41[1] - right_eye_39[1]

        lEAR = (right_vert_dist_1 + right_vert_dist_2)/(2*horizontal_dist_right)

        horizontal_dist_left = left_eye_left_point[0] - left_eye_right_point[0]
        left_vert_dist_1 = left_eye_47[1] - left_eye_45[1]
        left_vert_dist_2 = left_eye_48[1] - left_eye_44[1]

        rEAR = (left_vert_dist_1 + left_vert_dist_2)/(2*horizontal_dist_left)

        cv.line(frame, right_eye_right_point, right_eye_left_point,
                (0, 255, 0), 1)
        cv.line(frame, right_top_mid, right_bottom_mid, (0, 255, 0), 1)
        cv.line(frame, left_eye_right_point, left_eye_left_point,
                (0, 255, 0), 1)
        cv.line(frame, left_top_mid, left_bottom_mid, (0, 255, 0), 1)

        # Calculate YAWN aspect ratio
        right_mouth = (landmarks.part(60).x, landmarks.part(60).y)
        left_mouth = (landmarks.part(64).x, landmarks.part(64).y)

        top_mouth = (landmarks.part(62).x, landmarks.part(62).y)
        bottom_mouth = (landmarks.part(66).x, landmarks.part(66).y)

        mouthAR = (bottom_mouth[1]-top_mouth[1])/(left_mouth[0]-right_mouth[0])

        tEAR = round(lEAR + rEAR, 3)
        print('EAR total', tEAR)
        print("mouthAR ", round(mouthAR, 3))
    #begin timer of eyes being shut and counter for drowsy frames
    if tEAR <= 0.43 and start == 0:
        start = time.time()
        currFrames = 0
    #once eyes open restart the start and add to total drowsy frames
    elif tEAR > 0.43:
        start = 0
        drowsyFrames += currDrowsy
    #if eyes shut get the time elapsed
    else:
        currTime = time.time()
        elapsed = currTime-start
        print(elapsed)
        if elapsed > 3:
            #add one drowsy frame per loop once the 75 is counted
            if currFrames > 0:
                currFrames += 1
            #once elapsed at 3 seconds, driver has been drowsy for 75 seconds (25fps)
            else:
                currFrames += 75
            cv.putText(frame, "EAR: {}".format(tEAR), (10, 30),
                    cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv.putText(frame, "MAR: {:.2f}".format(round(mouthAR, 3)), (10, 60),
                    cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('framer', frame)
    if cv.waitKey(20) == ord('q'):
        print('stop')
        break

print('Drowsy Percentage: ', drowsyFrames/totalFrames)

cv.destroyAllWindows()
cap.release()
