import cv2
import mediapipe as mp
import time

class BodyDetector():
    def __init__(self, mode=False, maxBody=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5 ):
        self.mode = mode
        self.maxBody = maxBody
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findBody(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    self.mpDraw.DrawingSpec(
                        color=(0, 0, 255),
                        thickness=2,
                        circle_radius=4
                    )
                )

        return img

    def findPosition(self, img, bodyNo=0, draw= True):

        lmList = []

        if self.results.pose_landmarks:

            myBody = self.results.pose_landmarks

            for id, lm in enumerate(myBody.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([id, cx, cy])

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = BodyDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findBody(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()



    