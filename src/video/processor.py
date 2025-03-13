"""
Video processing module for weightlifting performance analysis
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Generator, Optional


class VideoProcessor:
    """
    Handles video loading, preprocessing, and frame extraction
    for weightlifting performance analysis.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize the VideoProcessor with the path to a video file.
        
        Args:
            video_path (str): Path to the video file to process
        """
        self.video_path = video_path
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.video_metadata = {}
        
    def load_video(self) -> Dict[str, Any]:
        """
        Load the video file and extract metadata.
        
        Returns:
            Dict[str, Any]: Video metadata including dimensions, FPS, and frame count
        """
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")
            
        # Extract video metadata
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.video_metadata = {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration_seconds': self.frame_count / self.fps if self.fps > 0 else 0,
            'format': self._get_video_format()
        }
        
        return self.video_metadata
        
    def _get_video_format(self) -> str:
        """
        Determine the format/resolution of the video.
        
        Returns:
            str: Description of the video format (e.g., "4K", "1080p", etc.)
        """
        if self.width >= 3840 and self.height >= 2160:
            return "4K"
        elif self.width >= 2560 and self.height >= 1440:
            return "1440p"
        elif self.width >= 1920 and self.height >= 1080:
            return "1080p"
        elif self.width >= 1280 and self.height >= 720:
            return "720p"
        else:
            return f"{self.width}x{self.height}"
    
    def frame_generator(self, skip_frames: int = 0) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generate frames from the video.
        
        Args:
            skip_frames (int, optional): Number of frames to skip between yields. Defaults to 0.
        
        Yields:
            Tuple[int, np.ndarray]: Frame number and the frame image
        """
        if self.cap is None or not self.cap.isOpened():
            self.load_video()
            
        frame_num = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            if frame_num % (skip_frames + 1) == 0:
                yield frame_num, frame
                
            frame_num += 1
            
    def resize_frame(self, frame: np.ndarray, target_width: int = 640) -> np.ndarray:
        """
        Resize a frame while maintaining aspect ratio.
        
        Args:
            frame (np.ndarray): Input frame
            target_width (int, optional): Desired width. Defaults to 640.
        
        Returns:
            np.ndarray: Resized frame
        """
        h, w = frame.shape[:2]
        ratio = target_width / w
        target_height = int(h * ratio)
        
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def release(self) -> None:
        """
        Release video capture resources
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def __enter__(self):
        """Context manager entry"""
        self.load_video()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()