# Weightlifting Performance Analyzer

A Python application that processes weightlifting videos to automatically analyze and provide performance metrics such as rep count, velocity, force applied, and power output.

## Features

- **Video Upload & Format Support**: Upload videos in various formats and resolutions including 4K60fps
- **User Input Parameters**: Input barbell weight and user height/weight for accurate performance analysis
- **Pose Estimation & Motion Tracking**: Detect key points on the user's body using MediaPipe
- **Automatic Rep Counting**: Accurately count repetitions performed during exercises
- **Performance Metrics Calculation**:
  - Velocity: Measure movement speed during exercises (concentric/eccentric phases)
  - Force Applied: Estimate force exerted based on movement and weights
  - Power Output (Watts): Compute power output combining velocity and force
- **Data Visualization**: 
  - Real-time metrics display
  - Phase-separated analysis (concentric/eccentric)
  - Dark mode support for reduced eye strain
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

2. Install using pip:
   ```bash
   pip install .
   ```

## Usage

### Graphical User Interface (GUI)

1. Launch the application:
   ```bash
   weightlifting-analyzer  # Normal mode
   weightlifting-analyzer --dark-mode  # Dark theme
   ```

2. Use the GUI to:
   - Open a weightlifting video
   - Set exercise parameters (type, barbell weight, user measurements)
   - Run the analysis
   - View performance metrics and visualizations
   - Save results

### Features Guide

- **Dark Mode**: Launch with `--dark-mode` for a dark theme that reduces eye strain
- **Real-time Metrics**: Key performance indicators are displayed at the top of the analysis
- **Phase Analysis**: All plots clearly show concentric (up) and eccentric (down) phases
- **Mean Values**: Automatic calculation and display of mean values for each metric
- **Quick Export**: Save both raw data (CSV) and visualizations (PNG) with one click

## Technical Details

- **Computer Vision**: Uses OpenCV and MediaPipe for pose detection and tracking
- **Physics Calculations**: Applies biomechanical principles to calculate performance metrics
- **Data Processing**: Uses NumPy, Pandas for data manipulation and analysis
- **Visualization**: Matplotlib and Seaborn for creating visual representations
- **User Interface**: Modern Qt-based interface with dark mode support

## Requirements

See `requirements.txt` for detailed dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.