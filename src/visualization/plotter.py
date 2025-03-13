"""
Data visualization module for weightlifting performance analysis
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import io
from matplotlib.figure import Figure


class PerformanceVisualizer:
    """
    Creates visualizations for weightlifting performance data including
    angles, velocity, force, and power metrics.
    """
    
    def __init__(self):
        """Initialize the PerformanceVisualizer"""
        # Set up a nice style for plots
        sns.set_theme(style="whitegrid")
        self.color_palette = sns.color_palette("muted")
    
    def plot_joint_angles(self, angle_data: pd.DataFrame, selected_angles: List[str] = None) -> Figure:
        """
        Plot joint angle changes over time.
        
        Args:
            angle_data (pd.DataFrame): DataFrame with time and angle data
            selected_angles (List[str], optional): List of angle names to plot
            
        Returns:
            Figure: Matplotlib figure with the plot
        """
        if angle_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No angle data available", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # If no specific angles selected, display the most important ones
        if not selected_angles:
            important_angles = ['avg_knee', 'avg_hip', 'left_knee', 'right_knee']
            selected_angles = [col for col in important_angles if col in angle_data.columns]
            
            # If none of the important angles are present, just select the first few
            if not selected_angles and len(angle_data.columns) > 1:
                selected_angles = [col for col in angle_data.columns if col != 'time'][:3]
        
        # Plot each selected angle
        for i, angle_name in enumerate(selected_angles):
            if angle_name in angle_data.columns:
                ax.plot(angle_data['time'], angle_data[angle_name], 
                       label=angle_name.replace('_', ' ').title(),
                       color=self.color_palette[i % len(self.color_palette)])
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Joint Angles Over Time')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_velocity(self, velocity_data: pd.DataFrame) -> Figure:
        """
        Plot velocity over time.
        
        Args:
            velocity_data (pd.DataFrame): DataFrame with time and velocity data
            
        Returns:
            Figure: Matplotlib figure with the plot
        """
        if velocity_data.empty or 'bar_velocity' not in velocity_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No velocity data available", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(velocity_data['time'], velocity_data['bar_velocity'], 
               label='Bar Velocity', color=self.color_palette[0])
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Bar Velocity Over Time')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_force(self, force_data: pd.DataFrame) -> Figure:
        """
        Plot force over time.
        
        Args:
            force_data (pd.DataFrame): DataFrame with time and force data
            
        Returns:
            Figure: Matplotlib figure with the plot
        """
        if force_data.empty or 'bar_force' not in force_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No force data available", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(force_data['time'], force_data['bar_force'], 
               label='Applied Force', color=self.color_palette[1])
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Applied Force Over Time')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_power(self, power_data: pd.DataFrame) -> Figure:
        """
        Plot power output over time.
        
        Args:
            power_data (pd.DataFrame): DataFrame with time and power data
            
        Returns:
            Figure: Matplotlib figure with the plot
        """
        if power_data.empty or 'bar_power' not in power_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No power data available", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(power_data['time'], power_data['bar_power'], 
               label='Power Output', color=self.color_palette[2])
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Output Over Time')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self, 
                               summary_metrics: Dict[str, Any], 
                               time_series_data: Dict[str, pd.DataFrame]) -> Figure:
        """
        Create a comprehensive dashboard with summary metrics and key graphs.
        
        Args:
            summary_metrics (Dict[str, Any]): Dictionary of summary metrics
            time_series_data (Dict[str, pd.DataFrame]): Dictionary of time series data
            
        Returns:
            Figure: Matplotlib figure with the dashboard
        """
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Weightlifting Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Create a grid for the plots
        gs = fig.add_gridspec(3, 3)
        
        # Summary metrics in the top left
        ax_summary = fig.add_subplot(gs[0, 0])
        self._plot_summary_text(ax_summary, summary_metrics)
        
        # Joint angles
        if 'angles' in time_series_data and not time_series_data['angles'].empty:
            ax_angles = fig.add_subplot(gs[0, 1:])
            self._plot_joint_angles_subplot(ax_angles, time_series_data['angles'])
        
        # Velocity
        if 'velocities' in time_series_data and not time_series_data['velocities'].empty:
            ax_velocity = fig.add_subplot(gs[1, :])
            self._plot_velocity_subplot(ax_velocity, time_series_data['velocities'])
        
        # Force and Power
        if 'forces' in time_series_data and not time_series_data['forces'].empty:
            ax_force = fig.add_subplot(gs[2, :2])
            self._plot_force_subplot(ax_force, time_series_data['forces'])
            
        if 'powers' in time_series_data and not time_series_data['powers'].empty:
            ax_power = fig.add_subplot(gs[2, 2])
            self._plot_power_subplot(ax_power, time_series_data['powers'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
        
        return fig
    
    def _plot_summary_text(self, ax, summary_metrics: Dict[str, Any]):
        """Helper method to plot summary metrics as text"""
        ax.axis('off')
        
        text_items = [
            f"Repetitions: {summary_metrics.get('rep_count', 'N/A')}",
            f"Max Velocity: {summary_metrics.get('max_velocity', 'N/A'):.2f} {summary_metrics.get('max_velocity_units', 'm/s')}",
            f"Avg Velocity: {summary_metrics.get('avg_velocity', 'N/A'):.2f} {summary_metrics.get('avg_velocity_units', 'm/s')}",
            f"Max Force: {summary_metrics.get('max_force', 'N/A'):.1f} {summary_metrics.get('max_force_units', 'N')}",
            f"Avg Force: {summary_metrics.get('avg_force', 'N/A'):.1f} {summary_metrics.get('avg_force_units', 'N')}",
            f"Max Power: {summary_metrics.get('max_power', 'N/A'):.1f} {summary_metrics.get('max_power_units', 'W')}",
            f"Avg Power: {summary_metrics.get('avg_power', 'N/A'):.1f} {summary_metrics.get('avg_power_units', 'W')}"
        ]
        
        # Create a nice summary box
        summary_text = '\n'.join(text_items)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=props)
        ax.set_title('Performance Summary', fontweight='bold')
    
    def _plot_joint_angles_subplot(self, ax, angle_data: pd.DataFrame):
        """Helper method to plot joint angles on a subplot"""
        important_angles = ['avg_knee', 'avg_hip']
        selected_angles = [col for col in important_angles if col in angle_data.columns]
        
        if not selected_angles and len(angle_data.columns) > 1:
            selected_angles = [col for col in angle_data.columns if col != 'time'][:2]
            
        for i, angle_name in enumerate(selected_angles):
            if angle_name in angle_data.columns:
                ax.plot(angle_data['time'], angle_data[angle_name], 
                       label=angle_name.replace('_', ' ').title(),
                       color=self.color_palette[i % len(self.color_palette)])
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Joint Angles')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, alpha=0.3)
    
    def _plot_velocity_subplot(self, ax, velocity_data: pd.DataFrame):
        """Helper method to plot velocity on a subplot"""
        if 'bar_velocity' in velocity_data.columns:
            ax.plot(velocity_data['time'], velocity_data['bar_velocity'], 
                   label='Velocity', color=self.color_palette[0])
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title('Bar Velocity')
            ax.grid(True, alpha=0.3)
    
    def _plot_force_subplot(self, ax, force_data: pd.DataFrame):
        """Helper method to plot force on a subplot"""
        if 'bar_force' in force_data.columns:
            ax.plot(force_data['time'], force_data['bar_force'], 
                   label='Force', color=self.color_palette[1])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Force (N)')
            ax.set_title('Applied Force')
            ax.grid(True, alpha=0.3)
    
    def _plot_power_subplot(self, ax, power_data: pd.DataFrame):
        """Helper method to plot power on a subplot"""
        if 'bar_power' in power_data.columns:
            ax.plot(power_data['time'], power_data['bar_power'], 
                   label='Power', color=self.color_palette[2])
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power (W)')
            ax.set_title('Power Output')
            ax.grid(True, alpha=0.3)
            
    def save_figure_to_buffer(self, fig: Figure, format: str = 'png', dpi: int = 100) -> bytes:
        """
        Save a matplotlib figure to a bytes buffer.
        
        Args:
            fig (Figure): Matplotlib figure
            format (str): Image format (e.g., 'png', 'pdf')
            dpi (int): DPI for rasterized formats
            
        Returns:
            bytes: Figure data as bytes
        """
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi)
        buf.seek(0)
        return buf.getvalue()