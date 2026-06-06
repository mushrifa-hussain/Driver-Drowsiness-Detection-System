from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound


# Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

COUNTER = 0

print("[INFO] Loading facial landmark predictor...")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

# Eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] Starting webcam...")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    for face in faces:

        # Face rectangle
        x = face.left()
        y = face.top()
        w = face.right()
        h = face.bottom()

        cv2.rectangle(
            frame,
            (x, y),
            (w, h),
            (255, 0, 0),
            2
        )

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Status
        status = "AWAKE"
        color = (0, 255, 0)

        if ear < EAR_THRESHOLD:
            COUNTER += 1
            status = "DROWSY"
            color = (0, 0, 255)

            if COUNTER >= CONSEC_FRAMES:

                cv2.putText(
                    frame,
                    "DRIVER DROWSINESS DETECTED",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                )

                winsound.Beep(2500, 500)

        else:
            COUNTER = 0

        # Status text
        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # EAR display
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (500, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()