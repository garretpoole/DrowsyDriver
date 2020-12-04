import cv2 as cv
import dlib
import numpy as np
import sys
import time
import imutils


def PUC_calc(p1, p2, p3, p4, p5, p6):
    area = ((np.linalg.norm([p2 - p5])) / 2) ** 2
    perimeter = np.linalg.norm(p1 - p2) + np.linalg.norm(p2 - p3) + np.linalg.norm(p3 - p4) + np.linalg.norm(
        p4 - p5) + np.linalg.norm(p5 - p6) + np.linalg.norm(p6 - p1)
    return (4 * np.pi * area) / (perimeter ** 2)


def drowsy_score(EAR, MAR, PUC, MOE):
    score = 0
    if EAR < 0.43:
        score += 1.5
    if MAR > 0.5:
        score += 1
    if PUC < 0.1:
        score += 1
    if MOE > 1.2:
        score += 1
    return score


def get_baseline(cap):
    MOE, PUC, tEAR, mouthAR = 0, 0, 0, 0
    count = 0
    cv.namedWindow('Initialization')
    while True:
        count += 1
        ret, frame = cap.read()
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

            left_eye_right_point = (landmarks.part(42).x, landmarks.part(42).y)
            left_eye_left_point = (landmarks.part(45).x, landmarks.part(45).y)

            left_eye_44 = (landmarks.part(43).x, landmarks.part(43).y)
            left_eye_48 = (landmarks.part(47).x, landmarks.part(47).y)

            left_eye_45 = (landmarks.part(44).x, landmarks.part(44).y)
            left_eye_47 = (landmarks.part(46).x, landmarks.part(46).y)

            horizontal_dist_right = right_eye_left_point[0] - \
                                    right_eye_right_point[0]
            right_vert_dist_1 = right_eye_42[1] - right_eye_38[1]
            right_vert_dist_2 = right_eye_41[1] - right_eye_39[1]

            lEAR = (right_vert_dist_1 + right_vert_dist_2) / (2 * horizontal_dist_right)

            horizontal_dist_left = left_eye_left_point[0] - left_eye_right_point[0]
            left_vert_dist_1 = left_eye_47[1] - left_eye_45[1]
            left_vert_dist_2 = left_eye_48[1] - left_eye_44[1]

            rEAR = (left_vert_dist_1 + left_vert_dist_2) / (2 * horizontal_dist_left)

            # Calculate YAWN aspect ratio
            right_mouth = (landmarks.part(60).x, landmarks.part(60).y)
            left_mouth = (landmarks.part(64).x, landmarks.part(64).y)

            top_mouth = (landmarks.part(62).x, landmarks.part(62).y)
            bottom_mouth = (landmarks.part(66).x, landmarks.part(66).y)

            mouthAR += (bottom_mouth[1] - top_mouth[1]) / (left_mouth[0] - right_mouth[0])
            rPUC = PUC_calc(np.array(right_eye_right_point), np.array(right_eye_38), np.array(right_eye_39),
                            np.array(right_eye_left_point), np.array(right_eye_41), np.array(right_eye_42))
            lPUC = PUC_calc(np.array(left_eye_right_point), np.array(left_eye_44), np.array(left_eye_45),
                            np.array(left_eye_left_point), np.array(left_eye_47), np.array(left_eye_48))
            PUC += (lPUC + rPUC) / 2
            tEAR += round(lEAR + rEAR, 3)

            cv.putText(frame, "Measuring...", (10, 30),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv.imshow("Initialization", frame)
            if cv.waitKey(20) == ord('q'):
                print('stop')
                break
            # Mouth over Ear aspect ratio
            MOE += mouthAR / tEAR
            if count >= 25:
                cv.destroyWindow("Initialization")
                return tEAR / 25, mouthAR / 25, MOE / 25, PUC / 25


if len(sys.argv) != 2:
    print('argument length=', len(sys.argv))
    print("usage: python3 [script name] [video file name or 0 for camera input]")
    exit(0)

file = sys.argv[1]
if file == '0':
    file = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
input("Press Enter to begin measurement...")
cap = cv.VideoCapture(file)

# Collect 25 frames of sample data to get baseline alert state
baselines = get_baseline(cap)
EAR_base, MAR_base, MOE_base, PUC_base = baselines[0], baselines[1], baselines[2], baselines[3]
# print(baselines)


cv.namedWindow('framer')
closed_eye_count = 0
counter = 0
score = [0] * 45
while True:

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    if not ret:
        print('read not successful')
        break
    tEAR, mouthAR, MOE, PUC = 0, 0, 0, 0
    frame = imutils.resize(frame, width=450)
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

        right_top_mid = ((right_eye_39[0] + right_eye_38[0]) // 2,
                         (right_eye_38[1] + right_eye_39[1]) // 2)
        right_bottom_mid = ((right_eye_41[0] + right_eye_42[0]) // 2,
                            (right_eye_41[1] + right_eye_42[1]) // 2)

        left_eye_right_point = (landmarks.part(42).x, landmarks.part(42).y)
        left_eye_left_point = (landmarks.part(45).x, landmarks.part(45).y)

        left_eye_44 = (landmarks.part(43).x, landmarks.part(43).y)
        left_eye_48 = (landmarks.part(47).x, landmarks.part(47).y)

        left_eye_45 = (landmarks.part(44).x, landmarks.part(44).y)
        left_eye_47 = (landmarks.part(46).x, landmarks.part(46).y)

        left_top_mid = ((left_eye_44[0] + left_eye_45[0]) // 2,
                        (left_eye_44[1] + left_eye_45[1]) // 2)
        left_bottom_mid = ((left_eye_47[0] + left_eye_48[0]) // 2,
                           (left_eye_47[1] + left_eye_48[1]) // 2)

        horizontal_dist_right = right_eye_left_point[0] - \
                                right_eye_right_point[0]
        right_vert_dist_1 = right_eye_42[1] - right_eye_38[1]
        right_vert_dist_2 = right_eye_41[1] - right_eye_39[1]

        lEAR = (right_vert_dist_1 + right_vert_dist_2) / (2 * horizontal_dist_right)

        horizontal_dist_left = left_eye_left_point[0] - left_eye_right_point[0]
        left_vert_dist_1 = left_eye_47[1] - left_eye_45[1]
        left_vert_dist_2 = left_eye_48[1] - left_eye_44[1]

        rEAR = (left_vert_dist_1 + left_vert_dist_2) / (2 * horizontal_dist_left)

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

        mouthAR = (bottom_mouth[1] - top_mouth[1]) / (left_mouth[0] - right_mouth[0])
        print(right_eye_right_point, right_eye_38, right_eye_39, right_eye_left_point, right_eye_41, right_eye_42)
        rPUC = PUC_calc(np.array(right_eye_right_point), np.array(right_eye_38), np.array(right_eye_39),
                        np.array(right_eye_left_point), np.array(right_eye_41), np.array(right_eye_42))
        lPUC = PUC_calc(np.array(left_eye_right_point), np.array(left_eye_44), np.array(left_eye_45),
                        np.array(left_eye_left_point), np.array(left_eye_47), np.array(left_eye_48))
        PUC = (lPUC + rPUC) / 2
        tEAR = round(lEAR + rEAR, 3)
        print('EAR total', tEAR)
        print("mouthAR ", round(mouthAR, 3))

        # Mouth over Ear aspect ratio
        MOE = mouthAR / tEAR
    if tEAR < 0.43:
        print(time.time())
        closed_eye_count += 1
    else:
        closed_eye_count = 0
    score[counter % 45] = drowsy_score(tEAR, mouthAR, PUC, MOE)
    counter += 1
    cv.putText(frame, "EAR: {}".format(tEAR), (10, 30),
               cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, "MAR: {:.2f}".format(round(mouthAR, 3)), (10, 60),
               cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, "MOE: {:.2f}".format(round(MOE, 3)), (10, 90),
               cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, "PUC: {:.2f}".format(round(PUC, 3)), (10, 120),
               cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, "Score: {:.2f}".format(round(sum(score) / 45, 3)), (10, 150),
               cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    if sum(score) / 45 > 2.2:
        cv.putText(frame, "DROWSY ALERT!", (10, 180), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        print(time.time())
    cv.imshow('framer', frame)

    if cv.waitKey(20) == ord('q'):
        print('stop')
        break

cv.destroyAllWindows()
cap.release()
