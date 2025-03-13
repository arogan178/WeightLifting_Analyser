"""
Performance metrics calculation module for weightlifting analysis
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


class PerformanceCalculator:
    """
    Calculates performance metrics for weightlifting exercises including
    rep count, velocity, force, and power output.
    """
    
    def __init__(self, user_weight: float = 0, barbell_weight: float = 0, user_height: float = 0):
        """
        Initialize the PerformanceCalculator with user parameters.
        
        Args:
            user_weight (float): User's weight in kg (optional)
            barbell_weight (float): Weight of the barbell in kg
            user_height (float): User's height in cm (optional)
        """
        self.user_weight = user_weight
        self.barbell_weight = barbell_weight
        self.user_height = user_height
        self.gravity = 9.81  # m/s^2
        
        # For storing time-series data
        self.joint_angle_series = {}
        self.position_series = {}
        self.velocity_series = {}
        self.acceleration_series = {}
        self.force_series = {}
        self.power_series = {}
        
    def set_parameters(self, user_weight: float = None, barbell_weight: float = None, user_height: float = None):
        """
        Update user parameters.
        
        Args:
            user_weight (float, optional): User's weight in kg
            barbell_weight (float, optional): Weight of the barbell in kg
            user_height (float, optional): User's height in cm
        """
        if user_weight is not None:
            self.user_weight = user_weight
            
        if barbell_weight is not None:
            self.barbell_weight = barbell_weight
            
        if user_height is not None:
            self.user_height = user_height
    
    def process_frame_data(self, 
                          frame_number: int,
                          frame_time: float,
                          pose_data: Dict[str, Any], 
                          exercise_type: str = 'squat'):
        """
        Process pose data from a single frame and update time-series data.
        
        Args:
            frame_number (int): Sequential frame number
            frame_time (float): Time in seconds for this frame
            pose_data (Dict): Pose data from the pose estimation module
            exercise_type (str): Type of exercise being performed
        """
        if not pose_data.get('joint_angles') or not pose_data.get('landmarks'):
            return
        
        # Store joint angles over time
        angles = pose_data['joint_angles']
        for angle_name, angle_value in angles.items():
            if angle_name not in self.joint_angle_series:
                self.joint_angle_series[angle_name] = []
            self.joint_angle_series[angle_name].append((frame_time, angle_value))
        
        # Calculate bar position based on exercise type
        bar_position = self._calculate_bar_position(pose_data, exercise_type)
        
        # Store bar position over time
        if 'bar_height' not in self.position_series:
            self.position_series['bar_height'] = []
        self.position_series['bar_height'].append((frame_time, bar_position))
        
        # If we have at least 3 data points, we can calculate velocity and other metrics
        if len(self.position_series['bar_height']) >= 3:
            self._update_velocity(frame_time)
            self._update_acceleration(frame_time)
            self._update_force(frame_time)
            self._update_power(frame_time)
    
    def _calculate_bar_position(self, pose_data: Dict[str, Any], exercise_type: str) -> float:
        """
        Calculate the bar position based on exercise type and pose landmarks.
        
        Args:
            pose_data (Dict): Pose data from the pose estimation module
            exercise_type (str): Type of exercise being performed
            
        Returns:
            float: Height position of the barbell (normalized)
        """
        landmarks = pose_data['landmarks']
        
        if exercise_type.lower() == 'squat':
            # For squats, we track the bar at shoulder level
            shoulder_y = (landmarks[11]['y'] + landmarks[12]['y']) / 2
            return shoulder_y
            
        elif exercise_type.lower() == 'deadlift':
            # For deadlifts, track the bar at wrist/hand level
            wrist_y = (landmarks[15]['y'] + landmarks[16]['y']) / 2
            return wrist_y
            
        elif exercise_type.lower() == 'bench_press':
            # For bench press, track the bar at wrist/arm level
            wrist_y = (landmarks[15]['y'] + landmarks[16]['y']) / 2
            return wrist_y
        
        # Default to shoulder height for other exercises
        shoulder_y = (landmarks[11]['y'] + landmarks[12]['y']) / 2
        return shoulder_y
    
    def _update_velocity(self, current_time: float):
        """
        Calculate velocity based on position changes.
        Uses centered finite difference method for better accuracy.
        
        Args:
            current_time (float): Current frame time
        """
        positions = self.position_series['bar_height']
        
        # Need at least 3 points for centered difference
        if len(positions) < 3:
            return
        
        # Get the latest 3 position readings
        times = [p[0] for p in positions[-3:]]
        pos = [p[1] for p in positions[-3:]]
        
        # Calculate velocity using centered finite difference
        # v = (x_{i+1} - x_{i-1}) / (t_{i+1} - t_{i-1})
        dt = times[2] - times[0]
        
        if dt > 0:
            velocity = (pos[0] - pos[2]) / dt  # Note: y-axis is inverted in image coords
            
            if 'bar_velocity' not in self.velocity_series:
                self.velocity_series['bar_velocity'] = []
            
            self.velocity_series['bar_velocity'].append((current_time, velocity))
    
    def _update_acceleration(self, current_time: float):
        """
        Calculate acceleration based on velocity changes.
        
        Args:
            current_time (float): Current frame time
        """
        if 'bar_velocity' not in self.velocity_series or len(self.velocity_series['bar_velocity']) < 3:
            return
        
        velocities = self.velocity_series['bar_velocity']
        
        # Get the latest 3 velocity readings
        times = [v[0] for v in velocities[-3:]]
        vels = [v[1] for v in velocities[-3:]]
        
        # Calculate acceleration using centered finite difference
        dt = times[2] - times[0]
        
        if dt > 0:
            acceleration = (vels[2] - vels[0]) / dt
            
            if 'bar_acceleration' not in self.acceleration_series:
                self.acceleration_series['bar_acceleration'] = []
            
            self.acceleration_series['bar_acceleration'].append((current_time, acceleration))
    
    def _update_force(self, current_time: float):
        """
        Calculate force using F = m·a.
        
        Args:
            current_time (float): Current frame time
        """
        if 'bar_acceleration' not in self.acceleration_series or not self.acceleration_series['bar_acceleration']:
            return
        
        # Mass is barbell weight plus a portion of user weight depending on the exercise
        mass = self.barbell_weight  # kg
        
        # Get the latest acceleration
        acceleration = self.acceleration_series['bar_acceleration'][-1][1]
        
        # Calculate force: F = ma
        force = mass * acceleration
        
        # Add the weight force (mass * g)
        force += mass * self.gravity
        
        if 'bar_force' not in self.force_series:
            self.force_series['bar_force'] = []
        
        self.force_series['bar_force'].append((current_time, force))
    
    def _update_power(self, current_time: float):
        """
        Calculate power using P = F·v.
        
        Args:
            current_time (float): Current frame time
        """
        if ('bar_force' not in self.force_series or not self.force_series['bar_force'] or
            'bar_velocity' not in self.velocity_series or not self.velocity_series['bar_velocity']):
            return
        
        # Get the latest force and velocity
        force = self.force_series['bar_force'][-1][1]
        velocity = self.velocity_series['bar_velocity'][-1][1]
        
        # Calculate power: P = F·v (watts)
        power = force * velocity
        
        if 'bar_power' not in self.power_series:
            self.power_series['bar_power'] = []
        
        self.power_series['bar_power'].append((current_time, power))
    
    def _get_rep_segments(self, angle_name: str = 'avg_knee', exercise_type: str = 'squat') -> List[Tuple[int, int]]:
        """
        Get start and end indices for each repetition.
        
        Returns:
            List[Tuple[int, int]]: List of (start_idx, end_idx) for each rep
        """
        if angle_name not in self.joint_angle_series or len(self.joint_angle_series[angle_name]) < 10:
            return []
        
        angle_data = [a[1] for a in self.joint_angle_series[angle_name]]
        
        try:
            # Smooth the data
            window_size = min(51, len(angle_data) - 2 if len(angle_data) % 2 == 0 else len(angle_data) - 1)
            if window_size < 3:
                window_size = 3
            if window_size % 2 == 0:
                window_size -= 1
            
            smoothed_data = savgol_filter(angle_data, window_size, 3)
            
            # Find peaks and valleys
            if exercise_type.lower() == 'squat':
                valleys, _ = find_peaks(-np.array(smoothed_data))
                peaks, _ = find_peaks(smoothed_data)
            else:
                peaks, _ = find_peaks(smoothed_data)
                valleys, _ = find_peaks(-np.array(smoothed_data))
            
            # Sort all points
            all_points = sorted(list(peaks) + list(valleys))
            
            # Create segments
            segments = []
            for i in range(len(all_points) - 1):
                segments.append((all_points[i], all_points[i + 1]))
                
            return segments
            
        except Exception as e:
            print(f"Error in rep segmentation: {e}")
            return []

    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for the exercise session.
        
        Returns:
            Dict[str, Any]: Dictionary with summary metrics
        """
        summary = {}
        
        # Get rep segments
        segments = self._get_rep_segments()
        summary['rep_count'] = len(segments)
        
        # Calculate per-rep metrics
        rep_velocities = []
        rep_forces = []
        rep_powers = []
        
        for start_idx, end_idx in segments:
            # Velocity calculations per rep
            if 'bar_velocity' in self.velocity_series:
                velocities = [v[1] for v in self.velocity_series['bar_velocity'][start_idx:end_idx]]
                if velocities:
                    rep_velocities.append(np.mean([abs(v) for v in velocities]))  # Use absolute values
                    
            # Force calculations per rep
            if 'bar_force' in self.force_series:
                forces = [f[1] for f in self.force_series['bar_force'][start_idx:end_idx]]
                if forces:
                    rep_forces.append(np.mean([abs(f) for f in forces]))
                    
            # Power calculations per rep
            if 'bar_power' in self.power_series:
                powers = [p[1] for p in self.power_series['bar_power'][start_idx:end_idx]]
                positive_powers = [p for p in powers if p > 0]
                if positive_powers:
                    rep_powers.append(np.mean(positive_powers))
        
        # Calculate averages across reps
        summary['avg_velocity'] = np.mean(rep_velocities) if rep_velocities else 0
        summary['avg_force'] = np.mean(rep_forces) if rep_forces else 0
        summary['avg_power'] = np.mean(rep_powers) if rep_powers else 0
        
        # Calculate maximums from entire series
        if 'bar_velocity' in self.velocity_series and self.velocity_series['bar_velocity']:
            summary['max_velocity'] = max(abs(v[1]) for v in self.velocity_series['bar_velocity'])
        else:
            summary['max_velocity'] = 0
            
        if 'bar_force' in self.force_series and self.force_series['bar_force']:
            summary['max_force'] = max(abs(f[1]) for f in self.force_series['bar_force'])
        else:
            summary['max_force'] = 0
            
        if 'bar_power' in self.power_series and self.power_series['bar_power']:
            summary['max_power'] = max(p[1] for p in self.power_series['bar_power'] if p[1] > 0)
        else:
            summary['max_power'] = 0
        
        # Add units
        summary['avg_velocity_units'] = 'm/s'
        summary['max_velocity_units'] = 'm/s'
        summary['avg_force_units'] = 'N'
        summary['max_force_units'] = 'N'
        summary['avg_power_units'] = 'W'
        summary['max_power_units'] = 'W'
        
        return summary
    
    def get_time_series_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get all time series data as pandas DataFrames for further analysis or visualization.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames for each metric type
        """
        results = {}
        
        # Convert joint angle series to DataFrame
        if self.joint_angle_series:
            angle_data = {}
            for angle_name, series in self.joint_angle_series.items():
                times, values = zip(*series) if series else ([], [])
                angle_data[angle_name] = values
                if 'time' not in angle_data:
                    angle_data['time'] = times
            results['angles'] = pd.DataFrame(angle_data)
        
        # Convert position series to DataFrame
        if self.position_series:
            pos_data = {}
            for pos_name, series in self.position_series.items():
                times, values = zip(*series) if series else ([], [])
                pos_data[pos_name] = values
                if 'time' not in pos_data:
                    pos_data['time'] = times
            results['positions'] = pd.DataFrame(pos_data)
        
        # Convert velocity, acceleration, force, and power series to DataFrames
        for name, data_dict in [
            ('velocities', self.velocity_series),
            ('accelerations', self.acceleration_series),
            ('forces', self.force_series),
            ('powers', self.power_series)
        ]:
            if data_dict:
                temp_data = {}
                for metric_name, series in data_dict.items():
                    times, values = zip(*series) if series else ([], [])
                    temp_data[metric_name] = values
                    if 'time' not in temp_data:
                        temp_data['time'] = times
                results[name] = pd.DataFrame(temp_data)
        
        return results