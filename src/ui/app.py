"""
User interface module for weightlifting performance analyzer using PySide6
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox,
    QDoubleSpinBox, QTabWidget, QSplitter, QFrame, QGridLayout,
    QGroupBox, QScrollArea, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QGuiApplication
try:
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg
except ImportError:
    # Fallback to Qt5 backend if Qt6 is not available
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.video.processor import VideoProcessor
from src.pose.estimator import PoseEstimator
from src.metrics.calculator import PerformanceCalculator
from src.visualization.plotter import PerformanceVisualizer

class VideoWidget(QLabel):
    """Widget for displaying video frames"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 225)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #212121; color: white; }")
        self.setText("No video loaded")
        
    def set_frame(self, frame):
        """Display a frame in the widget"""
        if frame is None:
            self.setText("Invalid frame")
            return
            
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create QImage from the frame
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Get widget size
            widget_size = self.size()
            
            # Scale maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(image).scaled(
                widget_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
                
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.setText(f"Error displaying frame: {str(e)}")
            raise

class ParametersWidget(QWidget):
    """Widget for input parameters"""
    
    # Update the signal definition to match the actual parameters being emitted
    parameters_changed = Signal(str, float, float, float)  # exercise, barbell, height, weight
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Exercise type
        exercise_group = QGroupBox("Exercise Type")
        exercise_layout = QVBoxLayout()
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems(["squat", "deadlift", "bench_press"])
        exercise_layout.addWidget(self.exercise_combo)
        exercise_group.setLayout(exercise_layout)
        layout.addWidget(exercise_group)
        
        # Weights and measurements
        metrics_group = QGroupBox("Parameters")
        metrics_layout = QGridLayout()
        
        # Barbell weight
        metrics_layout.addWidget(QLabel("Barbell Weight (kg):"), 0, 0)
        self.barbell_weight = QDoubleSpinBox()
        self.barbell_weight.setRange(0, 500)
        self.barbell_weight.setValue(200)
        self.barbell_weight.setSingleStep(2.5)
        metrics_layout.addWidget(self.barbell_weight, 0, 1)
        
        # User height
        metrics_layout.addWidget(QLabel("User Height (cm):"), 1, 0)
        self.user_height = QDoubleSpinBox()
        self.user_height.setRange(100, 250)
        self.user_height.setValue(175)
        self.user_height.setSingleStep(1)
        metrics_layout.addWidget(self.user_height, 1, 1)
        
        # User weight
        metrics_layout.addWidget(QLabel("User Weight (kg):"), 2, 0)
        self.user_weight = QDoubleSpinBox()
        self.user_weight.setRange(30, 200)
        self.user_weight.setValue(75)
        self.user_weight.setSingleStep(1)
        metrics_layout.addWidget(self.user_weight, 2, 1)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Connect signals
        self.exercise_combo.currentTextChanged.connect(self._parameters_changed)
        self.barbell_weight.valueChanged.connect(self._parameters_changed)
        self.user_height.valueChanged.connect(self._parameters_changed)
        self.user_weight.valueChanged.connect(self._parameters_changed)
        
        layout.addStretch()
        
    def _parameters_changed(self):
        """Emit signal when any parameter changes"""
        self.parameters_changed.emit(
            self.exercise_combo.currentText(),
            self.barbell_weight.value(),
            self.user_height.value(),
            self.user_weight.value()
        )
        
    def get_parameters(self):
        """Get current parameter values"""
        return {
            'exercise_type': self.exercise_combo.currentText(),
            'barbell_weight': self.barbell_weight.value(),
            'user_height': self.user_height.value(),
            'user_weight': self.user_weight.value()
        }

class ResultsWidget(QTabWidget):
    """Widget for displaying analysis results"""
    
    def __init__(self, parent=None, dark_mode=False):
        super().__init__(parent)
        self.visualizer = PerformanceVisualizer(dark_mode=dark_mode)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_ui()
        
    def _setup_ui(self):
        # Summary tab
        self.summary_tab = QScrollArea()
        self.summary_tab.setWidgetResizable(True)
        self.summary_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.summary_content = QWidget()
        self.summary_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.summary_layout = QVBoxLayout(self.summary_content)
        
        # Add placeholder for summary dashboard
        self.dashboard_frame = QWidget()
        self.dashboard_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dashboard_layout = QVBoxLayout(self.dashboard_frame)
        self.summary_layout.addWidget(self.dashboard_frame)
        
        self.summary_tab.setWidget(self.summary_content)
        self.addTab(self.summary_tab, "Summary")
        
    def update_results(self, summary_results, time_series_data):
        """Update the display with new results"""
        # Clear previous dashboard
        for i in reversed(range(self.dashboard_layout.count())): 
            self.dashboard_layout.itemAt(i).widget().setParent(None)

        if not time_series_data:
            return
            
        # Summary dashboard with increased size
        summary_fig = self.visualizer.create_summary_dashboard(
            time_series_data,
            fig_width=16,
            fig_height=10
        )
        summary_canvas = FigureCanvasQTAgg(summary_fig)
        summary_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        summary_canvas.setMinimumSize(QSize(800, 600))
        self.dashboard_layout.addWidget(summary_canvas)
        
        # Add stretch to ensure the canvas fills the space
        self.dashboard_layout.addStretch(1)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, dark_mode=False):
        super().__init__()
        self.setWindowTitle("Weightlifting Performance Analyzer")
        self.resize(1400, 900)
        
        # Instance variables
        self.dark_mode = dark_mode
        self.video_path = None
        self.video_processor = None
        self.pose_estimator = None
        self.performance_calculator = None
        self.processing_timer = None
        self.current_frame = None
        self.is_processing = False
        self.visualizer = PerformanceVisualizer(dark_mode=dark_mode)
        
        self._setup_ui()
        self._setup_theme()
        
    def _setup_theme(self):
        """Apply dark/light theme to the application"""
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2d2d2d;
                    color: white;
                }
                QPushButton {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #555555;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QPushButton:pressed {
                    background-color: #353535;
                }
                QProgressBar {
                    border: 1px solid #555555;
                    border-radius: 3px;
                    background-color: #404040;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #2ca02c;
                }
                QGroupBox {
                    border: 1px solid #555555;
                    margin-top: 0.5em;
                    color: white;
                }
                QGroupBox::title {
                    color: white;
                }
                QComboBox, QDoubleSpinBox {
                    background-color: #404040;
                    color: white;
                    border: 1px solid #555555;
                    padding: 3px;
                }
            """)
        else:
            self.setStyleSheet("")  # Use default light theme
        
    def _setup_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top section with video and parameters
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left side - Video and controls in a vertical layout
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Control buttons on top of video
        controls = QHBoxLayout()
        self.open_btn = QPushButton("Open Video")
        self.analyze_btn = QPushButton("Analyze")
        self.stop_btn = QPushButton("Stop")
        
        controls.addWidget(self.open_btn)
        controls.addWidget(self.analyze_btn)
        controls.addWidget(self.stop_btn)
        
        # Progress bar
        self.progress = QProgressBar()
        controls.addWidget(self.progress)
        
        video_layout.addLayout(controls)
        
        # Video display
        self.video_widget = VideoWidget()
        video_layout.addWidget(self.video_widget)
        
        # Add video panel to top section
        top_layout.addWidget(video_panel)
        
        # Parameters panel on right
        self.parameters = ParametersWidget()
        top_layout.addWidget(self.parameters)
        
        # Set stretch factors to make video smaller (1/4)
        top_layout.setStretchFactor(video_panel, 1)
        top_layout.setStretchFactor(self.parameters, 3)
        
        # Add top section to main layout
        main_layout.addWidget(top_section)
        
        # Results section
        self.results = ResultsWidget(dark_mode=self.dark_mode)
        main_layout.addWidget(self.results)
        
        # Save button at the bottom
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setFixedWidth(120)  # Set fixed width for the button
        save_layout.addWidget(self.save_btn)
        save_layout.addStretch()
        main_layout.addLayout(save_layout)

        # Connect signals
        self.open_btn.clicked.connect(self._open_video)
        self.analyze_btn.clicked.connect(self._start_analysis)
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.save_btn.clicked.connect(self._save_results)
        
        # Initial button states
        self.analyze_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        # Set stretch factors to make bottom section take up more space
        main_layout.setStretchFactor(top_section, 1)
        main_layout.setStretchFactor(self.results, 5)  # Increased from 3 to 5
        
    def _open_video(self):
        """Open a video file"""
        try:
            # Clean up any existing video resources
            if self.video_processor:
                self.video_processor.release()
                self.video_processor = None
            
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Video",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
            )
            
            if not file_path:
                return
                
            self.video_path = file_path
            
            # Load video and show first frame
            self.video_processor = VideoProcessor(file_path)
            metadata = self.video_processor.load_video()
            
            if metadata['frame_count'] == 0:
                raise ValueError("Video appears to be empty")
            
            # Get first frame
            frame_data = next(self.video_processor.frame_generator())
            if frame_data is None or frame_data[1] is None:
                raise ValueError("Could not read first frame of video")
                
            self.video_widget.set_frame(frame_data[1])
            
            # Enable analyze button
            self.analyze_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to load video: {str(e)}\nPlease make sure the video file is valid and not corrupted."
            )
            self.video_path = None
            self.analyze_btn.setEnabled(False)
            if self.video_processor:
                self.video_processor.release()
                self.video_processor = None
            
    def _start_analysis(self):
        """Start video analysis"""
        if self.is_processing or not self.video_path:
            return
            
        # Get parameters
        params = self.parameters.get_parameters()
        
        # Initialize components
        self.video_processor = VideoProcessor(self.video_path)
        self.pose_estimator = PoseEstimator()
        self.performance_calculator = PerformanceCalculator(
            user_weight=params['user_weight'],
            barbell_weight=params['barbell_weight'],
            user_height=params['user_height']
        )
        
        # Load video
        metadata = self.video_processor.load_video()
        self.total_frames = metadata['frame_count']
        self.current_frame = 0
        
        # Setup frame processing timer
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self._process_next_frame)
        
        # Update UI state
        self.is_processing = True
        self.analyze_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        
        # Start processing
        self.frame_generator = self.video_processor.frame_generator(
            skip_frames=1 if self.total_frames > 1000 else 0
        )
        self.processing_timer.start(1)  # Process frames as fast as possible
        
    def _process_next_frame(self):
        """Process the next video frame"""
        try:
            frame_num, frame = next(self.frame_generator)
            
            # Update progress
            progress = int((frame_num / self.total_frames) * 100)
            self.progress.setValue(progress)
            
            # Process frame
            annotated_frame, pose_data = self.pose_estimator.process_frame(frame)
            
            # Update display
            self.video_widget.set_frame(annotated_frame)
            
            # Calculate metrics
            frame_time = frame_num / self.video_processor.fps
            self.performance_calculator.process_frame_data(
                frame_num,
                frame_time,
                pose_data,
                self.parameters.get_parameters()['exercise_type']
            )
            
        except StopIteration:
            # Finished processing
            self._finish_analysis()

    def _finish_analysis(self):
        """Complete the analysis and update results"""
        self.processing_timer.stop()
        
        # Calculate final results
        summary_results = self.performance_calculator.calculate_summary_metrics()
        time_series_data = self.performance_calculator.get_time_series_data()
        
        # Store results for later use (e.g., saving)
        self.summary_results = summary_results
        self.time_series_data = time_series_data
        
        # Update results display
        self.results.update_results(summary_results, time_series_data)
        
        # Update UI state
        self.is_processing = False
        self.analyze_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
        # Clean up
        if self.pose_estimator:
            self.pose_estimator.release()
        if self.video_processor:
            self.video_processor.release()
            
    def _stop_analysis(self):
        """Stop the ongoing analysis"""
        if self.processing_timer:
            self.processing_timer.stop()
        self.is_processing = False
        self.analyze_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def _save_results(self):
        """Save analysis results"""
        if not hasattr(self, 'time_series_data'):
            return
            
        # Get save directory
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Results",
            ""
        )
        
        if not save_dir:
            return
            
        try:
            # Generate base filename
            base_name = Path(self.video_path).stem
            
            # Save CSV files
            for name, data in self.time_series_data.items():
                if data is not None and not data.empty:
                    csv_path = os.path.join(save_dir, f"{base_name}_{name}.csv")
                    data.to_csv(csv_path, index=False)
            
            # Save plots with larger size for better quality
            summary_fig = self.visualizer.create_summary_dashboard(
                self.time_series_data,
                fig_width=16, fig_height=10  # Increased from (12, 8) to (16, 10)
            )
            summary_fig.savefig(
                os.path.join(save_dir, f"{base_name}_summary.png"),
                dpi=150
            )
            
            # Save individual detailed plots
            if 'velocities' in self.time_series_data:
                velocity_fig = self.visualizer.plot_velocity(
                    self.time_series_data['velocities'],
                    fig_width=8, fig_height=5
                )
                velocity_fig.savefig(
                    os.path.join(save_dir, f"{base_name}_velocity.png"),
                    dpi=150
                )
            
            if 'powers' in self.time_series_data:
                power_fig = self.visualizer.plot_power(
                    self.time_series_data['powers'],
                    fig_width=8, fig_height=5
                )
                power_fig.savefig(
                    os.path.join(save_dir, f"{base_name}_power.png"),
                    dpi=150
                )
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Success",
                f"Results saved to {save_dir}"
            )
            
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save results: {str(e)}"
            )

def launch_app():
    """Launch the application"""
    import sys
    import os
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication
    
    # Set all Qt settings before creating QApplication
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create QApplication only after all settings are applied
    app = QApplication(sys.argv)
    
    # Check for dark mode argument
    dark_mode = '--dark-mode' in sys.argv
    
    window = MainWindow(dark_mode=dark_mode)
    window.show()
    
    return app.exec_()