"""
User interface module for weightlifting performance analyzer
"""
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from src.video.processor import VideoProcessor
from src.pose.estimator import PoseEstimator
from src.metrics.calculator import PerformanceCalculator
from src.visualization.plotter import PerformanceVisualizer


class PerformanceAnalyzerApp:
    """
    Main application class for the Weightlifting Performance Analyzer GUI
    """
    
    def __init__(self, root):
        """
        Initialize the application with a tkinter root window.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Weightlifting Performance Analyzer")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set application icon if available
        try:
            # You would need to create an icon file
            self.root.iconbitmap("resources/icon.ico")
        except:
            pass
        
        # Instance variables
        self.video_path = None
        self.exercise_type = tk.StringVar(value="Squat")
        self.barbell_weight = tk.DoubleVar(value=20)  # Default to standard Olympic bar
        self.user_height = tk.DoubleVar(value=175)    # Default height in cm
        self.user_weight = tk.DoubleVar(value=80)     # Default weight in kg
        
        # Video dimensions to instance variables
        self.video_width = None
        self.video_height = None
        
        # Processing objects
        self.video_processor = None
        self.pose_estimator = None
        self.performance_calculator = None
        self.visualizer = PerformanceVisualizer()
        
        # Processing state
        self.processing_thread = None
        self.is_processing = False
        self.stop_processing = False
        self.current_frame = None
        self.video_metadata = None
        self.summary_results = None
        self.time_series_data = None
        
        # Setup UI components
        self._create_menu()
        self._create_main_layout()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Video", command=self._open_video)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results", command=self._save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _create_main_layout(self):
        """Create the main application layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for video display and parameters
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video frame (left side of top frame)
        self.video_frame = ttk.LabelFrame(top_frame, text="Video", padding="5")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create two subframes for video and controls
        video_display_frame = ttk.Frame(self.video_frame)
        video_display_frame.pack(padx=5, pady=5)  # Remove fill and expand
        
        # Video display
        self.video_canvas = tk.Canvas(video_display_frame, bg="black", width=640, height=480)
        self.video_canvas.pack()  # Remove fill and expand to maintain fixed size
        
        # Video controls - pack directly in video_frame after video_display_frame
        video_controls = ttk.Frame(self.video_frame)
        video_controls.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.open_btn = ttk.Button(video_controls, text="Open Video", command=self._open_video)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = ttk.Button(video_controls, text="Analyze", command=self._start_analysis)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_btn.config(state=tk.DISABLED)
        
        self.stop_btn = ttk.Button(video_controls, text="Stop", command=self._stop_analysis)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Progress bar
        self.progress = ttk.Progressbar(video_controls, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Parameters frame (right side of top frame)
        params_frame = ttk.LabelFrame(top_frame, text="Parameters", padding="5", width=300)
        params_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)  # Add back the pack command
        params_frame.pack_propagate(False)
        
        # Exercise type
        exercise_frame = ttk.Frame(params_frame)
        exercise_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(exercise_frame, text="Exercise Type:").pack(side=tk.LEFT, padx=5)
        exercise_cb = ttk.Combobox(exercise_frame, textvariable=self.exercise_type,
                                values=["squat", "deadlift", "bench_press"])
        exercise_cb.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Barbell weight
        weight_frame = ttk.Frame(params_frame)
        weight_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(weight_frame, text="Barbell Weight (kg):").pack(side=tk.LEFT, padx=5)
        weight_entry = ttk.Entry(weight_frame, textvariable=self.barbell_weight, width=10)
        weight_entry.pack(side=tk.RIGHT, padx=5)
        
        # User height
        height_frame = ttk.Frame(params_frame)
        height_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(height_frame, text="User Height (cm):").pack(side=tk.LEFT, padx=5)
        height_entry = ttk.Entry(height_frame, textvariable=self.user_height, width=10)
        height_entry.pack(side=tk.RIGHT, padx=5)
        
        # User weight
        user_weight_frame = ttk.Frame(params_frame)
        user_weight_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(user_weight_frame, text="User Weight (kg):").pack(side=tk.LEFT, padx=5)
        user_weight_entry = ttk.Entry(user_weight_frame, textvariable=self.user_weight, width=10)
        user_weight_entry.pack(side=tk.RIGHT, padx=5)
        
        # Video metadata display
        self.metadata_text = tk.Text(params_frame, height=8, width=40, wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.X, expand=True, padx=5, pady=10)
        self.metadata_text.config(state=tk.DISABLED)
        
        # Bottom frame for analysis results
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create notebook for different results views
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_tab, text="Summary")
        
        # Create placeholder for the summary dashboard
        self.summary_placeholder = ttk.Frame(self.summary_tab)
        self.summary_placeholder.pack(fill=tk.BOTH, expand=True)
        
        # Graphs tab
        self.graphs_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.graphs_tab, text="Detailed Graphs")
        
        # Create tabs for different graph types
        self.graphs_notebook = ttk.Notebook(self.graphs_tab)
        self.graphs_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Angles tab
        self.angles_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.angles_tab, text="Joint Angles")
        
        # Velocity tab
        self.velocity_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.velocity_tab, text="Velocity")
        
        # Force tab
        self.force_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.force_tab, text="Force")
        
        # Power tab
        self.power_tab = ttk.Frame(self.graphs_notebook)
        self.graphs_notebook.add(self.power_tab, text="Power")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _display_frame(self, frame):
        """
        Display a frame on the video canvas at half resolution
        
        Args:
            frame: OpenCV frame (numpy array)
        """
        if frame is None or self.video_width is None or self.video_height is None:
            return
            
        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate half dimensions
        display_width = self.video_width // 2
        display_height = self.video_height // 2
        
        # Resize frame to half resolution
        display_frame = cv2.resize(frame_rgb, (display_width, display_height), 
                                 interpolation=cv2.INTER_AREA)
        
        # Convert to PhotoImage
        image = Image.fromarray(display_frame)
        photo = ImageTk.PhotoImage(image=image)
        
        # Clear previous content
        self.video_canvas.delete("all")
        
        # Update canvas size if needed
        if self.video_canvas.winfo_width() != display_width or self.video_canvas.winfo_height() != display_height:
            self.video_canvas.config(width=display_width, height=display_height)
        
        # Display image
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        # Keep a reference to prevent garbage collection
        self.current_frame = photo

    def _start_analysis(self):
        """Start video analysis in a separate thread"""
        if self.is_processing:
            messagebox.showinfo("Processing", "Analysis is already in progress")
            return
            
        if not self.video_path:
            messagebox.showwarning("No video", "Please open a video file first")
            return
            
        # Reset processing flags and UI
        self.is_processing = True
        self.stop_processing = False
        self.progress.config(value=0)
        
        # Disable analyze button, enable stop button
        self.analyze_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Update status
        self.status_var.set("Starting analysis...")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._run_analysis)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def _run_analysis(self):
        """Run the main analysis process in a background thread"""
        try:
            # Initialize components
            self.video_processor = VideoProcessor(self.video_path)
            self.pose_estimator = PoseEstimator()
            self.performance_calculator = PerformanceCalculator(
                user_weight=self.user_weight.get(),
                barbell_weight=self.barbell_weight.get(),
                user_height=self.user_height.get()
            )
            
            # Load video
            self.video_metadata = self.video_processor.load_video()
            total_frames = self.video_metadata['frame_count']
            
            # Process frame by frame
            exercise_type = self.exercise_type.get()
            
            # May need to skip some frames for performance
            skip_frames = 0
            if total_frames > 1000:
                skip_frames = 1  # Process every other frame for long videos
            
            for frame_num, frame in self.video_processor.frame_generator(skip_frames=skip_frames):
                if self.stop_processing:
                    break
                    
                # Update progress
                progress_value = int((frame_num / total_frames) * 100)
                self.root.after(0, lambda v=progress_value: self.progress.config(value=v))
                
                # Process frame with pose estimator
                annotated_frame, pose_data = self.pose_estimator.process_frame(frame)
                
                # Calculate performance metrics
                frame_time = frame_num / self.video_metadata['fps']
                self.performance_calculator.process_frame_data(
                    frame_num,
                    frame_time,
                    pose_data,
                    exercise_type
                )
                
                # Update UI with annotated frame
                self.root.after(0, lambda f=annotated_frame: self._display_frame(f))
                
                # Update status with frame info
                status_text = f"Processing frame {frame_num}/{total_frames} ({progress_value}%)"
                self.root.after(0, lambda t=status_text: self.status_var.set(t))
                
                # Small sleep to allow UI updates
                time.sleep(0.01)
            
            # Calculate final results
            self.summary_results = self.performance_calculator.calculate_summary_metrics()
            self.time_series_data = self.performance_calculator.get_time_series_data()
            
            # Update UI with results
            self.root.after(0, self._update_results)
            
            # Mark processing as complete
            self.is_processing = False
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Analysis complete"))
            
        except Exception as e:
            # Handle errors
            self.is_processing = False
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda e=str(e): self.status_var.set(f"Error: {e}"))
            self.root.after(0, lambda e=str(e): messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}"))
            
        finally:
            # Clean up
            if self.pose_estimator:
                self.pose_estimator.release()
            
            if self.video_processor:
                self.video_processor.release()
    
    def _update_results(self):
        """Update the UI with analysis results"""
        if not self.summary_results or not self.time_series_data:
            return
            
        # Create summary dashboard
        try:
            # Clear previous figures in summary tab
            for widget in self.summary_placeholder.winfo_children():
                widget.destroy()
                
            # Create and display summary dashboard
            fig = self.visualizer.create_summary_dashboard(
                self.summary_results,
                self.time_series_data
            )
            
            canvas = FigureCanvasTkAgg(fig, master=self.summary_placeholder)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Error updating summary dashboard: {e}")
        
        # Update detailed graph tabs
        self._update_graph_tab(self.angles_tab, 'angles', 
                             self.visualizer.plot_joint_angles, self.time_series_data.get('angles'))
        self._update_graph_tab(self.velocity_tab, 'velocity', 
                             self.visualizer.plot_velocity, self.time_series_data.get('velocities'))
        self._update_graph_tab(self.force_tab, 'force', 
                             self.visualizer.plot_force, self.time_series_data.get('forces'))
        self._update_graph_tab(self.power_tab, 'power', 
                             self.visualizer.plot_power, self.time_series_data.get('powers'))
    
    def _update_graph_tab(self, tab, name, plot_func, data):
        """
        Update a specific graph tab with visualization
        
        Args:
            tab: Tab frame to update
            name: Name of the data category
            plot_func: Function to call for plotting
            data: Data to plot
        """
        # Clear previous content
        for widget in tab.winfo_children():
            widget.destroy()
            
        if data is not None and not data.empty:
            try:
                # Create figure
                fig = plot_func(data)
                
                # Display in tab
                canvas = FigureCanvasTkAgg(fig, master=tab)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            except Exception as e:
                print(f"Error updating {name} tab: {e}")
                ttk.Label(tab, text=f"Error creating graph: {str(e)}").pack(pady=20)
        else:
            ttk.Label(tab, text=f"No {name} data available").pack(pady=20)
    
    def _stop_analysis(self):
        """Stop the ongoing analysis"""
        if self.is_processing:
            self.stop_processing = True
            self.status_var.set("Stopping analysis...")
    
    def _save_results(self):
        """Save the analysis results to files"""
        if not self.summary_results or not self.time_series_data:
            messagebox.showinfo("No Results", "No analysis results to save")
            return
            
        # Ask for directory to save results
        save_dir = filedialog.askdirectory(
            title="Select directory to save results",
            initialdir="/"
        )
        
        if not save_dir:
            return
            
        try:
            # Generate base filename from video name
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            
            # Save summary dashboard
            if self.summary_results:
                fig = self.visualizer.create_summary_dashboard(
                    self.summary_results,
                    self.time_series_data
                )
                fig.savefig(os.path.join(save_dir, f"{base_name}_summary.png"), dpi=150)
                plt.close(fig)
                
            # Save detailed graphs
            for name, data in self.time_series_data.items():
                if data is not None and not data.empty:
                    # Save the data as CSV
                    data.to_csv(os.path.join(save_dir, f"{base_name}_{name}.csv"), index=False)
                    
            # Save summary metrics as text
            with open(os.path.join(save_dir, f"{base_name}_metrics.txt"), 'w') as f:
                f.write("Weightlifting Performance Analysis\n")
                f.write("================================\n\n")
                f.write(f"Video: {os.path.basename(self.video_path)}\n")
                f.write(f"Exercise: {self.exercise_type.get()}\n")
                f.write(f"Barbell Weight: {self.barbell_weight.get()} kg\n")
                f.write(f"User Height: {self.user_height.get()} cm\n")
                f.write(f"User Weight: {self.user_weight.get()} kg\n\n")
                f.write("Performance Metrics\n")
                f.write("-----------------\n")
                for key, value in self.summary_results.items():
                    if not key.endswith('_units'):
                        units = self.summary_results.get(f"{key}_units", "")
                        if isinstance(value, float):
                            f.write(f"{key.replace('_', ' ').title()}: {value:.2f} {units}\n")
                        else:
                            f.write(f"{key.replace('_', ' ').title()}: {value} {units}\n")
            
            messagebox.showinfo("Success", f"Results saved to {save_dir}")
            self.status_var.set(f"Results saved to {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
            self.status_var.set("Error saving results")
    
    def _show_about(self):
        """Show the about dialog"""
        messagebox.showinfo(
            "About",
            "Weightlifting Performance Analyzer\n\n"
            "A Python application for analyzing weightlifting performance\n"
            "using computer vision and biomechanics principles.\n\n"
            "Provides detailed metrics including repetition count,\n"
            "velocity, force, and power output."
        )
        
    def _on_closing(self):
        """Handle window closing event"""
        if self.is_processing:
            if messagebox.askyesno("Quit", "Analysis is in progress. Are you sure you want to quit?"):
                self.stop_processing = True
                self.root.destroy()
        else:
            self.root.destroy()
    
    def _open_video(self):
            """Open a video file dialog and load the selected video"""
            filetypes = (
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            )
            
            filepath = filedialog.askopenfilename(
                title="Select a video file",
                initialdir="/",
                filetypes=filetypes
            )
            
            if not filepath:
                return
                
            self.video_path = filepath
            
            # Load video metadata
            try:
                self.video_processor = VideoProcessor(filepath)
                self.video_metadata = self.video_processor.load_video()
                
                # Store original video dimensions
                self.video_width = self.video_metadata['width']
                self.video_height = self.video_metadata['height']
                
                # Set canvas size to half the video dimensions
                self.video_canvas.config(width=self.video_width//2, height=self.video_height//2)
                
                # Update metadata information
                metadata_str = (
                    f"File: {os.path.basename(filepath)}\n"
                    f"Resolution: {self.video_metadata['width']}x{self.video_metadata['height']} "
                    f"({self.video_metadata['format']})\n"
                    f"FPS: {self.video_metadata['fps']:.1f}\n"
                    f"Duration: {self.video_metadata['duration_seconds']:.1f} seconds\n"
                    f"Total frames: {self.video_metadata['frame_count']}"
                )
                
                self.metadata_text.config(state=tk.NORMAL)
                self.metadata_text.delete(1.0, tk.END)
                self.metadata_text.insert(tk.END, metadata_str)
                self.metadata_text.config(state=tk.DISABLED)
                
                # Display the first frame
                self._display_frame(next(self.video_processor.frame_generator())[1])
                
                # Enable analyze button
                self.analyze_btn.config(state=tk.NORMAL)
                
                # Update status
                self.status_var.set(f"Loaded video: {os.path.basename(filepath)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video file: {str(e)}")
                self.status_var.set("Error loading video")
            

def launch_app(headless=False):
    """
    Launch the application.
    
    Args:
        headless (bool): If True, run in command line mode without GUI
    """
    if headless:
        # Implement command line version here
        print("Headless mode not implemented yet")
        return
        
    # Create and run GUI application
    root = tk.Tk()
    app = PerformanceAnalyzerApp(root)
    root.mainloop()