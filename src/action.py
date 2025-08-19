import cv2
import mediapipe as mp
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from action_classifier import PoseActionClassifier
from action_mapping import get_emotion_for_action

def main():
    parser = argparse.ArgumentParser(description="Real-Time Pose-Based Action Prediction Engine")
    parser.add_argument("--source", default="0", help="Video source (camera index or path to video file)")
    args = parser.parse_args()

    video_source = int(args.source) if args.source.isdigit() else args.source

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    classifier = PoseActionClassifier()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    print("Starting Action Prediction HUD...")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
