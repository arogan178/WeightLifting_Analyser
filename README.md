# Weightlifting Performance Analyzer

A Python application that processes weightlifting videos to automatically analyze and provide performance metrics such as rep count, velocity, force applied, and power output.

## Features

- **Video Upload & Format Support**: Upload videos in various formats and resolutions including 4K60fps
- **User Input Parameters**: Input barbell weight and user height/weight for accurate performance analysis
- **Pose Estimation & Motion Tracking**: Detect key points on the user's body using MediaPipe
- **Automatic Rep Counting**: Accurately count repetitions performed during exercises
- **Performance Metrics Calculation**:
  - Velocity: Measure movement speed during exercises
  - Force Applied: Estimate force exerted based on movement and weights
  - Power Output (Watts): Compute power output combining velocity and force
- **Data Visualization**: Present analysis results in user-friendly graphs and charts
- **Export Results**: Save analysis data in CSV format and graphs as images

## Supported Exercises

- Squats
- Deadlifts
- Bench Press

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/weightlifting-performance-analyzer.git
   cd weightlifting-performance-analyzer
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Graphical User Interface (GUI)

1. Launch the application:
   ```bash
   python main.py
   ```

2. Use the GUI to:
   - Open a weightlifting video
   - Set exercise parameters (type, barbell weight, user measurements)
   - Run the analysis
   - View performance metrics and visualizations
   - Save results

### Command Line (Coming Soon)

```bash
python main.py --no-gui --video=path/to/video.mp4 --exercise=squat --weight=100
```

## Project Structure

```
├── main.py                  # Main application entry point
├── requirements.txt         # Project dependencies
└── src/                     # Source code
    ├── video/               # Video processing
    │   └── processor.py     # Video frame extraction and preprocessing
    ├── pose/                # Pose estimation
    │   └── estimator.py     # Body keypoint detection and tracking
    ├── metrics/             # Performance metrics
    │   └── calculator.py    # Calculate rep count, velocity, force, power
    ├── visualization/       # Data visualization
    │   └── plotter.py       # Create graphs and charts
    └── ui/                  # User interface
        └── app.py           # GUI implementation
```

## Technical Details

- **Computer Vision**: Uses OpenCV and MediaPipe for pose detection and tracking
- **Physics Calculations**: Applies biomechanical principles to calculate performance metrics
- **Data Processing**: Uses NumPy, Pandas for data manipulation and analysis
- **Visualization**: Matplotlib and Seaborn for creating visual representations

## Requirements

See `requirements.txt` for detailed dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.