"""
Pose estimation module for weightlifting performance analysis using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Any, Optional


class PoseEstimator:
    """
    Handles pose estimation for weightlifting exercises using MediaPipe.
    Detects key points on the user's body and tracks movement patterns.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize the PoseEstimator with detection and tracking parameters.
        
        Args:
            min_detection_confidence (float): Minimum confidence for pose detection
            min_tracking_confidence (float): Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use the most accurate model
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Key joint indices for various exercises
        self.squat_joints = {
            'hip': [23, 24],          # Left and right hip
            'knee': [25, 26],         # Left and right knee
            'ankle': [27, 28],        # Left and right ankle
            'shoulder': [11, 12],     # Left and right shoulder
        }
        
        self.deadlift_joints = {
            'hip': [23, 24],          # Left and right hip
            'knee': [25, 26],         # Left and right knee
            'ankle': [27, 28],        # Left and right ankle
            'shoulder': [11, 12],     # Left and right shoulder
            'wrist': [15, 16],        # Left and right wrist
        }
        
        self.bench_press_joints = {
            'shoulder': [11, 12],     # Left and right shoulder
            'elbow': [13, 14],        # Left and right elbow
            'wrist': [15, 16],        # Left and right wrist
            'hip': [23, 24],          # Left and right hip
        }
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a video frame to detect body pose.
        
        Args:
            frame (np.ndarray): Input video frame
        
        Returns:
            Tuple[np.ndarray, Dict]: Annotated frame and pose data dictionary
        """
        # Convert BGR to RGB as MediaPipe requires RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.pose.process(frame_rgb)
        
        # Store pose data
        pose_data = self._extract_pose_data(results)
        
        # Draw the pose on the frame for visualization
        annotated_frame = self._draw_pose(frame.copy(), results)
        
        return annotated_frame, pose_data
    
    def _extract_pose_data(self, results) -> Dict[str, Any]:
        """
        Extract relevant pose data from the MediaPipe results.
        
        Args:
            results: MediaPipe pose processing results
            
        Returns:
            Dict[str, Any]: Structured pose data for analysis
        """
        if not results.pose_landmarks:
            return {'landmarks': None, 'joint_angles': None, 'visibility': None}
        
        # Extract landmark coordinates
        landmarks = []
        visibility = []
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
            })
            visibility.append(landmark.visibility)
        
        # Calculate important joint angles for exercises
        joint_angles = self._calculate_joint_angles(landmarks)
        
        return {
            'landmarks': landmarks,
            'joint_angles': joint_angles,
            'visibility': visibility
        }
    
    def _calculate_joint_angles(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate joint angles relevant for weightlifting exercises.
        
        Args:
            landmarks (List[Dict]): List of landmark coordinates
            
        Returns:
            Dict[str, float]: Dictionary of joint angles in degrees
        """
        angles = {}
        
        # Calculate knee angles (important for squats)
        angles['left_knee'] = self._calculate_angle(
            landmarks[self.squat_joints['hip'][0]],
            landmarks[self.squat_joints['knee'][0]],
            landmarks[self.squat_joints['ankle'][0]]
        )
        
        angles['right_knee'] = self._calculate_angle(
            landmarks[self.squat_joints['hip'][1]],
            landmarks[self.squat_joints['knee'][1]],
            landmarks[self.squat_joints['ankle'][1]]
        )
        
        # Calculate hip angles
        angles['left_hip'] = self._calculate_angle(
            landmarks[self.squat_joints['shoulder'][0]],
            landmarks[self.squat_joints['hip'][0]],
            landmarks[self.squat_joints['knee'][0]]
        )
        
        angles['right_hip'] = self._calculate_angle(
            landmarks[self.squat_joints['shoulder'][1]],
            landmarks[self.squat_joints['hip'][1]],
            landmarks[self.squat_joints['knee'][1]]
        )
        
        # Calculate average knee and hip angles (helpful for symmetry analysis)
        angles['avg_knee'] = (angles['left_knee'] + angles['right_knee']) / 2
        angles['avg_hip'] = (angles['left_hip'] + angles['right_hip']) / 2
        
        return angles
    
    @staticmethod
    def _calculate_angle(a: Dict[str, float], b: Dict[str, float], c: Dict[str, float]) -> float:
        """
        Calculate the angle between three points in 2D space.
        
        Args:
            a (Dict): First point coordinates
            b (Dict): Second point coordinates (vertex of the angle)
            c (Dict): Third point coordinates
            
        Returns:
            float: Angle in degrees
        """
        # Convert to numpy arrays for easier vector operations
        a_vec = np.array([a['x'], a['y']])
        b_vec = np.array([b['x'], b['y']])
        c_vec = np.array([c['x'], c['y']])
        
        # Calculate vectors
        ba = a_vec - b_vec
        bc = c_vec - b_vec
        
        # Calculate angle using dot product and vector magnitudes
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
        
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def _draw_pose(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw the detected pose landmarks and connections on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            results: MediaPipe pose detection results
            
        Returns:
            np.ndarray: Frame with pose visualization
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Additionally, we can highlight specific joints important for the exercise
            # This will be implemented based on the exercise type
            
        return frame
    
    def detect_exercise_phase(self, joint_angles: Dict[str, float], exercise_type: str = 'squat') -> str:
        """
        Detect the current phase of a weightlifting exercise based on joint angles.
        
        Args:
            joint_angles (Dict[str, float]): Dictionary of calculated joint angles
            exercise_type (str): Type of exercise ('squat', 'deadlift', 'bench_press')
            
        Returns:
            str: Current exercise phase ('eccentric', 'concentric', 'static', or 'unknown')
        """
        if not joint_angles:
            return 'unknown'
            
        if exercise_type.lower() == 'squat':
            # For squats, we primarily look at knee angles
            if joint_angles['avg_knee'] < 120:  # Deep squat position
                return 'eccentric_end'  # Bottom of the squat
            elif joint_angles['avg_knee'] > 170:  # Standing position
                return 'concentric_end'  # Top of the squat
            else:
                return 'transition'  # Somewhere in between
        
        # Additional exercise types can be added here
            
        return 'unknown'
    
    def release(self):
        """
        Release resources used by the pose estimator
        """
        self.pose.close()