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
            user_weight (float): User's weight in kg (kept for display purposes only)
            barbell_weight (float): Weight of the barbell in kg (used in calculations)
            user_height (float): User's height in cm (kept for display purposes only)
        """
        # These parameters are currently kept for display purposes in the UI
        self.user_weight = user_weight  
        self.user_height = user_height
        
        # This parameter is used in force and power calculations
        self.barbell_weight = barbell_weight
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
        # Store the current exercise type for use in other methods
        self.current_exercise_type = exercise_type.lower()
        
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
        Calculate velocity using central difference method with height normalization.
        
        Args:
            current_time (float): Current frame time
        """
        positions = self.position_series['bar_height']
        
        # Need at least 3 points for central difference
        if len(positions) < 3:
            return
        
        # Use simple central difference for robustness
        times = [p[0] for p in positions[-3:]]
        pos = [p[1] for p in positions[-3:]]
        
        dt = times[2] - times[0]
        
        if dt > 0:
            velocity = (pos[0] - pos[2]) / dt  # y-axis is inverted in image coords
            
            # Normalize velocity based on user height if available
            # Taller users have longer range of motion, affecting velocity measurements
            if self.user_height > 0:
                reference_height = 175  # cm, average height
                height_factor = self.user_height / reference_height
                velocity = velocity * height_factor  # Normalize by height ratio
            
            if 'bar_velocity' not in self.velocity_series:
                self.velocity_series['bar_velocity'] = []
            
            self.velocity_series['bar_velocity'].append((current_time, velocity))

    def _update_acceleration(self, current_time: float):
        """
        Calculate acceleration using central difference method.
        
        Args:
            current_time (float): Current frame time
        """
        if 'bar_velocity' not in self.velocity_series or len(self.velocity_series['bar_velocity']) < 3:
            return
        
        velocities = self.velocity_series['bar_velocity']
        
        # Use simple central difference for robustness
        times = [v[0] for v in velocities[-3:]]
        vels = [v[1] for v in velocities[-3:]]
        
        dt = times[2] - times[0]
        
        if dt > 0:
            acceleration = (vels[2] - vels[0]) / dt
            
            if 'bar_acceleration' not in self.acceleration_series:
                self.acceleration_series['bar_acceleration'] = []
            
            self.acceleration_series['bar_acceleration'].append((current_time, acceleration))
    
    def _update_force(self, current_time: float):
        """
        Calculate force with improved biomechanical model that includes user weight.
        
        Args:
            current_time (float): Current frame time
        """
        if 'bar_acceleration' not in self.acceleration_series or not self.acceleration_series['bar_acceleration']:
            return
        
        acceleration = self.acceleration_series['bar_acceleration'][-1][1]
        
        # Calculate effective mass based on exercise type
        # Include a portion of user weight based on exercise biomechanics
        exercise_type = getattr(self, 'current_exercise_type', 'squat').lower()
        
        # Movement-dependent factor for user weight contribution
        if exercise_type == 'squat':
            k_factor = 0.6  # ~60% of user weight contributes to squat
        elif exercise_type == 'deadlift':
            k_factor = 0.4  # ~40% of user weight contributes to deadlift
        elif exercise_type == 'bench_press':
            k_factor = 0.2  # ~20% of user weight contributes to bench press
        else:
            k_factor = 0.5  # Default factor
        
        # Calculate effective mass
        effective_mass = self.barbell_weight
        if self.user_weight > 0:
            effective_mass += k_factor * self.user_weight
        
        # Calculate force: F = ma + mg
        force = effective_mass * acceleration
        weight_force = effective_mass * self.gravity
        
        # Total force
        total_force = force + weight_force
        
        if 'bar_force' not in self.force_series:
            self.force_series['bar_force'] = []
        
        self.force_series['bar_force'].append((current_time, total_force))
    
    def _update_power(self, current_time: float):
        """
        Calculate power using P = F·v with improved biomechanical model.
        Only positive power (during concentric phase) is meaningful for performance analysis.
        
        Args:
            current_time (float): Current frame time
        """
        if ('bar_force' not in self.force_series or not self.force_series['bar_force'] or
            'bar_velocity' not in self.velocity_series or not self.velocity_series['bar_velocity']):
            return
        
        # Get the latest force and velocity
        force = self.force_series['bar_force'][-1][1]
        velocity = self.velocity_series['bar_velocity'][-1][1]
        
        # Calculate power: P = F·v
        power = force * velocity
        
        # Store power regardless of sign for complete analysis
        if 'bar_power' not in self.power_series:
            self.power_series['bar_power'] = []
        
        self.power_series['bar_power'].append((current_time, power))
    
    def _adaptive_threshold_detection(self, signal, min_distance=5, adaptive_ratio=0.1):
        """
        Perform adaptive threshold detection with personalized parameters
        based on user height and weight.
        
        Args:
            signal: The signal to analyze
            min_distance: Minimum distance between detections
            adaptive_ratio: Base ratio to be adjusted by user parameters
            
        Returns:
            (peaks, valleys): Indices of peaks and valleys
        """
        if len(signal) < 10:
            return [], []
        
        # Personalize adaptive_ratio based on user height and weight
        # Taller users tend to have slower, longer movements
        # Heavier users may have different movement patterns
        personalized_ratio = adaptive_ratio
        
        if self.user_height > 0 and self.user_weight > 0:
            # Calculate personalized threshold using the formula:
            # adaptive_ratio = 0.05 + 0.0003 × (height - 170) - 0.0002 × (weight - 75)
            height_adjustment = 0.0003 * (self.user_height - 170)
            weight_adjustment = 0.0002 * (self.user_weight - 75)
            personalized_ratio = 0.05 + height_adjustment - weight_adjustment
            
            # Ensure the ratio stays within reasonable bounds
            personalized_ratio = max(0.01, min(0.3, personalized_ratio))
        
        # Calculate prominence threshold with personalized ratio
        signal_range = np.max(signal) - np.min(signal)
        prominence_threshold = max(0.005, signal_range * personalized_ratio)
        
        # Adjust min_distance based on height (taller users = longer movements)
        if self.user_height > 0:
            height_factor = self.user_height / 175  # Relative to average height
            min_distance = max(3, int(min_distance * height_factor))
        
        # Detect peaks and valleys with personalized parameters
        peaks, _ = find_peaks(signal, 
                            distance=min_distance,
                            prominence=prominence_threshold,
                            width=2)
        
        valleys, _ = find_peaks(-np.array(signal), 
                              distance=min_distance,
                              prominence=prominence_threshold,
                              width=2)
        
        return peaks, valleys
    
    def _get_rep_segments(self, angle_name: str = 'avg_knee', exercise_type: str = 'squat') -> List[Tuple[int, int]]:
        """
        Get rep segments by analyzing velocity patterns.
        
        Args:
            angle_name (str): Not used, kept for backwards compatibility
            exercise_type (str): Type of exercise being performed
            
        Returns:
            List[Tuple[int, int]]: List of (start_idx, end_idx) for each rep
        """
        if 'bar_velocity' not in self.velocity_series or len(self.velocity_series['bar_velocity']) < 10:
            return []

        try:
            velocity_data = [v[1] for v in self.velocity_series['bar_velocity']]
            
            # Apply light smoothing
            window_size = min(max(5, len(velocity_data) // 30), 9)
            if window_size % 2 == 0:
                window_size -= 1
            
            smoothed_data = savgol_filter(velocity_data, window_size, 1)
            
            # Try adaptive peak detection first
            peaks, valleys = self._adaptive_threshold_detection(smoothed_data, min_distance=3, adaptive_ratio=0.1)
            
            # If we can't find enough peaks and valleys, try zero-crossing detection
            if len(peaks) < 2 or len(valleys) < 2:
                segments = []
                in_rep = False
                start_idx = 0
                
                for i in range(1, len(smoothed_data)):
                    # Detect start of rep (velocity becomes positive)
                    if smoothed_data[i-1] <= 0 and smoothed_data[i] > 0:
                        if not in_rep:
                            start_idx = i
                            in_rep = True
                    # Detect end of rep (velocity becomes negative)
                    elif smoothed_data[i-1] >= 0 and smoothed_data[i] < 0:
                        if in_rep:
                            segments.append((start_idx, i))
                            in_rep = False
                
                return segments
            
            # Use adaptive peak detection results
            segments = []
            valley_idx = 0
            peak_idx = 0

            while valley_idx < len(valleys) and peak_idx < len(peaks):
                valley = valleys[valley_idx]
                peak = peaks[peak_idx]

                if peak > valley:
                    # Validate minimum movement
                    segment = smoothed_data[valley:peak+1]
                    velocity_range = np.max(segment) - np.min(segment)
                    
                    # Lower threshold for validation
                    if velocity_range > 0.005:
                        segments.append((valley, peak))
                    
                    valley_idx += 1
                    peak_idx += 1
                elif peak < valley:
                    peak_idx += 1
                else:
                    valley_idx += 1

            return segments

        except Exception as e:
            print(f"Error in rep segmentation: {e}")
            return []

    def count_reps(self, angle_name: str = 'avg_knee', exercise_type: str = 'squat') -> int:
        """
        Count repetitions by analyzing velocity patterns.
        
        Args:
            exercise_type (str): Type of exercise
            
        Returns:
            int: Number of completed repetitions
        """
        # Simply call _get_rep_segments and count the results
        segments = self._get_rep_segments(angle_name, exercise_type)
        return len(segments)

    def calculate_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for the exercise session with separated 
        positive and negative velocity metrics.
        
        Returns:
            Dict[str, Any]: Dictionary with summary metrics
        """
        summary = {}
        
        # Get rep count using velocity-based detection
        summary['rep_count'] = self.count_reps()
        
        # Get rep segments for detailed metrics
        segments = self._get_rep_segments()
        
        # Calculate per-rep metrics
        rep_velocities_pos = []  # Positive velocities (concentric phase)
        rep_velocities_neg = []  # Negative velocities (eccentric phase)
        rep_forces = []
        rep_powers = []

        for start_idx, end_idx in segments:
            # Velocity calculations per rep, separated by direction
            if 'bar_velocity' in self.velocity_series:
                velocities = [v[1] for v in self.velocity_series['bar_velocity'][start_idx:end_idx]]
                if velocities:
                    # Separate positive and negative velocities
                    pos_vels = [v for v in velocities if v > 0]
                    neg_vels = [v for v in velocities if v < 0]
                    
                    if pos_vels:
                        rep_velocities_pos.append(np.mean(pos_vels))
                    
                    if neg_vels:
                        # Store negative velocities as absolute values for easier comparison
                        rep_velocities_neg.append(np.mean([abs(v) for v in neg_vels]))
                    
            # Force calculations per rep
            if 'bar_force' in self.force_series:
                forces = [f[1] for f in self.force_series['bar_force'][start_idx:end_idx]]
                if forces:
                    rep_forces.append(np.mean([abs(f) for f in forces]))
                    
            # Power calculations per rep - only for positive velocity (concentric) phases
            if ('bar_power' in self.power_series and 'bar_velocity' in self.velocity_series):
                # Get velocity and power values for this segment
                velocities = [v[1] for v in self.velocity_series['bar_velocity'][start_idx:end_idx]]
                powers = [p[1] for p in self.power_series['bar_power'][start_idx:end_idx]]
                
                # Pair velocities with their corresponding power values
                vel_power_pairs = list(zip(velocities, powers))
                
                # Only consider power during concentric phase (positive velocity)
                concentric_powers = [p for v, p in vel_power_pairs if v > 0]
                if concentric_powers:
                    rep_powers.append(np.mean(concentric_powers))

        # Calculate velocity metrics - separate positive and negative
        summary['avg_velocity_pos'] = np.mean(rep_velocities_pos) if rep_velocities_pos else 0
        summary['avg_velocity_neg'] = np.mean(rep_velocities_neg) if rep_velocities_neg else 0
        
        # Calculate force and power averages
        summary['avg_force'] = np.mean(rep_forces) if rep_forces else 0
        summary['avg_power'] = np.mean(rep_powers) if rep_powers else 0
        
        # Calculate maximums from entire series
        if 'bar_velocity' in self.velocity_series and self.velocity_series['bar_velocity']:
            all_velocities = [v[1] for v in self.velocity_series['bar_velocity']]
            pos_velocities = [v for v in all_velocities if v > 0]
            neg_velocities = [v for v in all_velocities if v < 0]
            
            summary['max_velocity_pos'] = max(pos_velocities) if pos_velocities else 0
            summary['max_velocity_neg'] = abs(min(neg_velocities)) if neg_velocities else 0
        else:
            summary['max_velocity_pos'] = 0
            summary['max_velocity_neg'] = 0
            
        if 'bar_force' in self.force_series and self.force_series['bar_force']:
            summary['max_force'] = max(abs(f[1]) for f in self.force_series['bar_force'])
        else:
            summary['max_force'] = 0
            
        # For power, only consider positive values during concentric movement
        if ('bar_power' in self.power_series and 'bar_velocity' in self.velocity_series):
            # Pair velocity and power from their series, using common timestamps
            vel_dict = {v[0]: v[1] for v in self.velocity_series['bar_velocity']}
            power_dict = {p[0]: p[1] for p in self.power_series['bar_power']}
            
            # Find timestamps present in both series
            common_times = set(vel_dict.keys()).intersection(set(power_dict.keys()))
            
            # Extract powers that correspond to positive velocity (concentric phase)
            concentric_powers = [power_dict[t] for t in common_times if vel_dict[t] > 0]
            
            if concentric_powers:
                summary['max_power'] = max(concentric_powers)
            else:
                summary['max_power'] = 0
        else:
            summary['max_power'] = 0
        
        # Add units
        summary['avg_velocity_pos_units'] = 'm/s'
        summary['max_velocity_pos_units'] = 'm/s'
        summary['avg_velocity_neg_units'] = 'm/s'
        summary['max_velocity_neg_units'] = 'm/s'
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