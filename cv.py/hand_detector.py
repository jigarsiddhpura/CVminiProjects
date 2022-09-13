import enum
import mediapipe as mp # library for 'noding' fingers - mapping finger joints
import cv2  

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
    #min_det.. checks for distance of hand from camera ... stop detecting if hand is too far or too close 
    
    #find hand nodes ka co ordinates
    def findLandmarks(self, image, handNumber=0, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        #using mediapipe chupa hua code
        results = self.hands.process(image)

        landMarkList = []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[handNumber]

            for id, landmark in enumerate(hand.landmark):
                imgH, imgW, imgC = originalImage.shape #dimensions of image
                xPos, yPos = int(landmark.x * imgW ) , int(landmark.y * imgH) 
                #landmarks are ratios - dimensions se multiply krke pixel perfect position of hand
                landMarkList.append([id, xPos, yPos])

            #drawing on pc
            if draw:
                mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)

        return landMarkList
