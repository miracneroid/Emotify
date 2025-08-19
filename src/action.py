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
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    classifier = PoseActionClassifier()
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    print("Starting Action Prediction HUD... Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        h, w, _ = frame.shape

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            prediction = classifier.classify(results.pose_landmarks.landmark, frame_shape=(h, w))
            action = prediction["action"]
            confidence = prediction["confidence"]
            details = prediction["details"]
            mapped_emotion = get_emotion_for_action(action)

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 140), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.rectangle(frame, (10, 10), (450, 140), (0, 255, 200), 2)

            cv2.putText(frame, f"Action: {action}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"Mapped Emotion: {mapped_emotion.upper()}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

            metrics_str = f"L-Arm: {details.get('l_arm_angle')} deg | R-Arm: {details.get('r_arm_angle')} deg"
            cv2.putText(frame, metrics_str, (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        else:
            cv2.putText(frame, "Searching for body pose...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Emotify - Action Prediction HUD", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
