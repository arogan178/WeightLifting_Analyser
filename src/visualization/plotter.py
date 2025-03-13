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
from matplotlib.patches import Rectangle


class PerformanceVisualizer:
    """
    Creates visualizations for weightlifting performance data including
    angles, velocity, force, and power metrics.
    """
    
    def __init__(self):
        """Initialize the PerformanceVisualizer"""
        # Set up a nice style for plots
        sns.set_theme(style="whitegrid")
        
        # Create a custom color palette for different metrics
        self.color_palette = {
            'velocity': '#1f77b4',  # Blue
            'velocity_pos': '#2ca02c',  # Green for concentric
            'velocity_neg': '#d62728',  # Red for eccentric
            'force': '#ff7f0e',     # Orange
            'power': '#9467bd',     # Purple
            'angles': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Varied colors for angles
        }
        
        # Set default plot style
        plt.rcParams.update({
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 12
        })
    
    def plot_joint_angles(self, angle_data: pd.DataFrame, selected_angles: List[str] = None, fig_width=5, fig_height=3) -> Figure:
        """
        Plot joint angle changes over time.
        
        Args:
            angle_data (pd.DataFrame): DataFrame with time and angle data
            selected_angles (List[str], optional): List of angle names to plot
            fig_width (float): Width of the figure in inches
            fig_height (float): Height of the figure in inches
           
        Returns:
            Figure: Matplotlib figure with the plot
        """
        if angle_data.empty:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.text(0.5, 0.5, "No angle data available", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
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
                       color=self.color_palette['angles'][i % len(self.color_palette['angles'])],
                       linewidth=1.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Joint Angles Over Time')
        
        # Add a nicer legend with smaller font
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
        
        # Enhance gridlines
        ax.grid(True, alpha=0.3, linestyle='-')
        
        # Add a light background color
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        return fig
    
    def plot_velocity(self, velocity_data: pd.DataFrame, fig_width=5, fig_height=3) -> Figure:
        """
        Plot velocity over time with concentric and eccentric phases highlighted.
        
        Args:
            velocity_data (pd.DataFrame): DataFrame with time and velocity data
            fig_width (float): Width of the figure in inches
            fig_height (float): Height of the figure in inches
           
        Returns:
            Figure: Matplotlib figure with the plot
        """
        if velocity_data.empty or 'bar_velocity' not in velocity_data.columns:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.text(0.5, 0.5, "No velocity data available", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        times = velocity_data['time'].values
        velocities = velocity_data['bar_velocity'].values
        
        # Calculate positive and negative velocities for different phases
        pos_mask = velocities > 0
        neg_mask = velocities < 0
        
        # Plot concentric phase (positive velocity)
        if np.any(pos_mask):
            ax.fill_between(times, velocities, 0, where=pos_mask, 
                          color=self.color_palette['velocity_pos'], alpha=0.3, 
                          label='Concentric Phase')
        
        # Plot eccentric phase (negative velocity)
        if np.any(neg_mask):
            ax.fill_between(times, velocities, 0, where=neg_mask, 
                          color=self.color_palette['velocity_neg'], alpha=0.3, 
                          label='Eccentric Phase')
        
        # Plot the velocity line
        ax.plot(times, velocities, 
               color=self.color_palette['velocity'], 
               linewidth=1.5, label='Bar Velocity')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=0.8)
        
        # Calculate separate means for positive and negative velocities
        if np.any(pos_mask):
            pos_mean = np.mean(velocities[pos_mask])
            ax.axhline(y=pos_mean, color=self.color_palette['velocity_pos'], 
                     linestyle='--', alpha=0.7, linewidth=1,
                     label=f'Mean Concentric: {pos_mean:.2f} m/s')
        
        if np.any(neg_mask):
            neg_mean = np.mean(velocities[neg_mask])
            ax.axhline(y=neg_mean, color=self.color_palette['velocity_neg'], 
                     linestyle='--', alpha=0.7, linewidth=1,
                     label=f'Mean Eccentric: {neg_mean:.2f} m/s')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Bar Velocity Over Time')
        
        # Add a nicer legend with smaller font
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
        
        # Set limit for better visualization if needed
        max_abs = max(abs(np.max(velocities)), abs(np.min(velocities))) if len(velocities) > 0 else 1
        y_limit = max(1, max_abs * 1.2)  # At least ±1 m/s
        ax.set_ylim(-y_limit, y_limit)
        
        # Enhance gridlines
        ax.grid(True, alpha=0.3, linestyle='-')
        
        # Add a light background color
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        return fig
    
    def plot_force(self, force_data: pd.DataFrame, fig_width=5, fig_height=3) -> Figure:
        """
        Plot force over time with visual enhancements.
        
        Args:
            force_data (pd.DataFrame): DataFrame with time and force data
            fig_width (float): Width of the figure in inches
            fig_height (float): Height of the figure in inches
           
        Returns:
            Figure: Matplotlib figure with the plot
        """
        if force_data.empty or 'bar_force' not in force_data.columns:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.text(0.5, 0.5, "No force data available", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        times = force_data['time'].values
        forces = force_data['bar_force'].values
        
        # Plot the force line with shaded area
        ax.plot(times, forces, color=self.color_palette['force'], 
               linewidth=1.5, label='Applied Force')
        ax.fill_between(times, forces, 0, color=self.color_palette['force'], alpha=0.2)
        
        # Add horizontal lines for mean and baseline
        if len(forces) > 0:
            mean_force = np.mean(forces)
            ax.axhline(y=mean_force, color=self.color_palette['force'], 
                     linestyle='--', alpha=0.7, linewidth=1, 
                     label=f'Mean Force: {mean_force:.1f} N')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Applied Force Over Time')
        
        # Add a nicer legend with smaller font
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
        
        # Enhance gridlines
        ax.grid(True, alpha=0.3, linestyle='-')
        
        # Add a light background color
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        return fig
    
    def plot_power(self, power_data: pd.DataFrame, fig_width=5, fig_height=3) -> Figure:
        """
        Plot power output over time with concentric phase highlighted.
        
        Args:
            power_data (pd.DataFrame): DataFrame with time and power data
            fig_width (float): Width of the figure in inches
            fig_height (float): Height of the figure in inches
           
        Returns:
            Figure: Matplotlib figure with the plot
        """
        if power_data.empty or 'bar_power' not in power_data.columns:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.text(0.5, 0.5, "No power data available", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        times = power_data['time'].values
        powers = power_data['bar_power'].values
        
        # If velocities DataFrame is included in power_data, use it to determine concentric phase
        if 'bar_velocity' in power_data.columns:
            velocities = power_data['bar_velocity'].values
            concentric_mask = velocities > 0  # Concentric phase = positive velocity
        else:
            # Otherwise just use positive power as a proxy for concentric phase
            concentric_mask = powers > 0
        
        # Highlight positive power (concentric phase) - most important for performance
        if np.any(concentric_mask):
            ax.fill_between(times, powers, 0, where=concentric_mask, 
                          color=self.color_palette['power'], alpha=0.3, 
                          label='Concentric Phase')
        
        # Plot the power line
        ax.plot(times, powers, color=self.color_palette['power'], 
               linewidth=1.5, label='Power Output')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=0.8)
        
        # Calculate and show mean for concentric phase power only
        if np.any(concentric_mask):
            # Only calculate mean for power values during concentric phase
            concentric_powers = powers[concentric_mask]
            mean_power = np.mean(concentric_powers)
            ax.axhline(y=mean_power, color=self.color_palette['power'], 
                     linestyle='--', alpha=0.7, linewidth=1,
                     label=f'Mean Power: {mean_power:.1f} W')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Output Over Time (Concentric Phase)')
        
        # Add a nicer legend with smaller font
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
        
        # Enhance gridlines
        ax.grid(True, alpha=0.3, linestyle='-')
        
        # Add a light background color
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self, 
                               summary_metrics: Dict[str, Any], 
                               time_series_data: Dict[str, pd.DataFrame],
                               fig_width=10, fig_height=8) -> Figure:
        """
        Create a comprehensive dashboard with summary metrics and key graphs.
        
        Args:
            summary_metrics (Dict[str, Any]): Dictionary of summary metrics
            time_series_data (Dict[str, pd.DataFrame]): Dictionary of time series data
            fig_width (float): Width of the figure in inches
            fig_height (float): Height of the figure in inches
           
        Returns:
            Figure: Matplotlib figure with the dashboard
        """
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        fig.suptitle('Weightlifting Performance Analysis', fontsize=14, fontweight='bold')
        
        # Create a grid for the plots
        gs = fig.add_gridspec(3, 3)
        
        # Summary metrics in the top left
        ax_summary = fig.add_subplot(gs[0, 0])
        self._plot_enhanced_summary(ax_summary, summary_metrics)
        
        # Joint angles
        if 'angles' in time_series_data and not time_series_data['angles'].empty:
            ax_angles = fig.add_subplot(gs[0, 1:])
            self._plot_joint_angles_subplot(ax_angles, time_series_data['angles'])
        
        # Velocity with concentric/eccentric phases
        if 'velocities' in time_series_data and not time_series_data['velocities'].empty:
            ax_velocity = fig.add_subplot(gs[1, :])
            self._plot_velocity_subplot_enhanced(ax_velocity, time_series_data['velocities'])
        
        # Force and Power
        if 'forces' in time_series_data and not time_series_data['forces'].empty:
            ax_force = fig.add_subplot(gs[2, :2])
            self._plot_force_subplot_enhanced(ax_force, time_series_data['forces'])
            
        if 'powers' in time_series_data and not time_series_data['powers'].empty:
            ax_power = fig.add_subplot(gs[2, 2])
            self._plot_power_subplot_enhanced(ax_power, time_series_data['powers'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
        
        return fig
    
    def _plot_enhanced_summary(self, ax, summary_metrics: Dict[str, Any]):
        """Helper method to plot summary metrics as text with enhanced styling"""
        ax.axis('off')
        
        # Extract and format metrics
        rep_count = summary_metrics.get('rep_count', 0)
        
        # Concentric (positive) velocity metrics
        max_vel_pos = summary_metrics.get('max_velocity_pos', 0)
        avg_vel_pos = summary_metrics.get('avg_velocity_pos', 0)
        vel_units = summary_metrics.get('max_velocity_pos_units', 'm/s')
        
        # Eccentric (negative) velocity metrics
        max_vel_neg = summary_metrics.get('max_velocity_neg', 0)
        avg_vel_neg = summary_metrics.get('avg_velocity_neg', 0)
        
        # Force metrics
        max_force = summary_metrics.get('max_force', 0)
        avg_force = summary_metrics.get('avg_force', 0)
        force_units = summary_metrics.get('max_force_units', 'N')
        
        # Power metrics
        max_power = summary_metrics.get('max_power', 0)
        avg_power = summary_metrics.get('avg_power', 0)
        power_units = summary_metrics.get('max_power_units', 'W')
        
        # Create formatted text sections
        text_sections = [
            "REPETITIONS\n" + f"{rep_count}",
            
            "CONCENTRIC (UP)\n" + 
            f"Max Velocity: {max_vel_pos:.2f} {vel_units}\n" +
            f"Avg Velocity: {avg_vel_pos:.2f} {vel_units}",
            
            "ECCENTRIC (DOWN)\n" + 
            f"Max Velocity: {max_vel_neg:.2f} {vel_units}\n" +
            f"Avg Velocity: {avg_vel_neg:.2f} {vel_units}",
            
            "FORCE\n" + 
            f"Maximum: {max_force:.1f} {force_units}\n" +
            f"Average: {avg_force:.1f} {force_units}",
            
            "POWER\n" + 
            f"Maximum: {max_power:.1f} {power_units}\n" +
            f"Average: {avg_power:.1f} {power_units}"
        ]
        
        # Define vertical positions for each section
        y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]
        
        # Plot each section with custom styling
        for i, (text, y_pos) in enumerate(zip(text_sections, y_positions)):
            # Extract the header (first line)
            lines = text.split('\n')
            header = lines[0]
            content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            # Draw header in bold
            ax.text(0.05, y_pos, header, transform=ax.transAxes, fontsize=9,
                   fontweight='bold', verticalalignment='top')
            
            # Draw content in normal font
            if content:
                ax.text(0.05, y_pos-0.05, content, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top')
        
        # Add a frame around the summary
        ax.set_frame_on(True)
        ax.patch.set_edgecolor('lightgray')
        ax.patch.set_facecolor('#f9f9f9')
        ax.patch.set_linewidth(1)
        ax.set_title('Performance Summary', fontweight='bold', fontsize=10)
    
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
                       color=self.color_palette['angles'][i],
                       linewidth=1.5)
        
        ax.set_xlabel('Time (s)', fontsize='small')
        ax.set_ylabel('Angle (degrees)', fontsize='small')
        ax.set_title('Joint Angles', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize='small')
        ax.legend(loc='best', fontsize='x-small', frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_facecolor('#f8f9fa')
    
    def _plot_velocity_subplot_enhanced(self, ax, velocity_data: pd.DataFrame):
        """Helper method to plot velocity with enhanced visuals on a subplot"""
        if 'bar_velocity' not in velocity_data.columns:
            ax.text(0.5, 0.5, "No velocity data available", ha='center', va='center')
            return
            
        times = velocity_data['time'].values
        velocities = velocity_data['bar_velocity'].values
        
        # Highlight positive and negative phases
        pos_mask = velocities > 0
        neg_mask = velocities < 0
        
        # Plot concentric phase (positive velocity)
        if np.any(pos_mask):
            ax.fill_between(times, velocities, 0, where=pos_mask, 
                          color=self.color_palette['velocity_pos'], alpha=0.3)
        
        # Plot eccentric phase (negative velocity)
        if np.any(neg_mask):
            ax.fill_between(times, velocities, 0, where=neg_mask, 
                          color=self.color_palette['velocity_neg'], alpha=0.3)
        
        # Plot the velocity line
        ax.plot(times, velocities, color=self.color_palette['velocity'], linewidth=1.5,
               label='Bar Velocity')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=0.8)
        
        # Calculate and add mean lines for each phase if data exists
        if np.any(pos_mask):
            pos_mean = np.mean(velocities[pos_mask])
            ax.axhline(y=pos_mean, color=self.color_palette['velocity_pos'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            ax.text(times[-1], pos_mean, f' {pos_mean:.2f}', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        if np.any(neg_mask):
            neg_mean = np.mean(velocities[neg_mask])
            ax.axhline(y=neg_mean, color=self.color_palette['velocity_neg'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            ax.text(times[-1], neg_mean, f' {neg_mean:.2f}', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        # Add legend with phase information
        legend_elements = [
            plt.Line2D([0], [0], color=self.color_palette['velocity'], lw=1.5, label='Bar Velocity'),
            plt.Rectangle((0, 0), 1, 1, fc=self.color_palette['velocity_pos'], alpha=0.3, label='Concentric (Up)'),
            plt.Rectangle((0, 0), 1, 1, fc=self.color_palette['velocity_neg'], alpha=0.3, label='Eccentric (Down)')
        ]
        
        ax.legend(handles=legend_elements, loc='best', fontsize=8, frameon=True, fancybox=True)
        
        ax.set_xlabel('Time (s)', fontsize='small')
        ax.set_ylabel('Velocity (m/s)', fontsize='small')
        ax.set_title('Bar Velocity', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize='small')
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_facecolor('#f8f9fa')
        
        # Add y-axis symmetry for better readability
        max_abs = max(abs(np.max(velocities)), abs(np.min(velocities))) if len(velocities) > 0 else 1
        y_limit = max(1, max_abs * 1.2)
        ax.set_ylim(-y_limit, y_limit)
    
    def _plot_force_subplot_enhanced(self, ax, force_data: pd.DataFrame):
        """Helper method to plot force with enhanced visuals on a subplot"""
        if 'bar_force' not in force_data.columns:
            ax.text(0.5, 0.5, "No force data available", ha='center', va='center')
            return
        
        times = force_data['time'].values
        forces = force_data['bar_force'].values
        
        # Plot the force line with shaded area
        ax.plot(times, forces, color=self.color_palette['force'], 
               linewidth=1.5, label='Force')
        ax.fill_between(times, forces, 0, color=self.color_palette['force'], alpha=0.2)
        
        # Add mean line
        if len(forces) > 0:
            mean_force = np.mean(forces)
            ax.axhline(y=mean_force, color=self.color_palette['force'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            ax.text(times[-1], mean_force, f' {mean_force:.1f}', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize='small')
        ax.set_ylabel('Force (N)', fontsize='small')
        ax.set_title('Applied Force', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize='small')
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_facecolor('#f8f9fa')
    
    def _plot_power_subplot_enhanced(self, ax, power_data: pd.DataFrame):
        """Helper method to plot power with enhanced visuals on a subplot"""
        if 'bar_power' not in power_data.columns:
            ax.text(0.5, 0.5, "No power data available", ha='center', va='center')
            return
        
        times = power_data['time'].values
        powers = power_data['bar_power'].values
        
        # Check if we have velocity data to determine concentric phase
        concentric_mask = None
        if 'bar_velocity' in power_data.columns:
            velocities = power_data['bar_velocity'].values
            concentric_mask = velocities > 0
        else:
            # Fall back to positive power as proxy for concentric phase
            concentric_mask = powers > 0
        
        # Highlight concentric phase (positive velocity)
        if np.any(concentric_mask):
            ax.fill_between(times, powers, 0, where=concentric_mask, 
                          color=self.color_palette['power'], alpha=0.3)
        
        # Plot the power line
        ax.plot(times, powers, color=self.color_palette['power'], linewidth=1.5, label='Power')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=0.8)
        
        # Calculate and add mean line for concentric phase power
        if np.any(concentric_mask):
            concentric_powers = powers[concentric_mask]
            pos_mean = np.mean(concentric_powers)
            ax.axhline(y=pos_mean, color=self.color_palette['power'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            ax.text(times[-1], pos_mean, f' {pos_mean:.1f}', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        ax.set_xlabel('Time (s)', fontsize='small')
        ax.set_ylabel('Power (W)', fontsize='small')
        ax.set_title('Power Output (Concentric Phase)', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize='small')
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_facecolor('#f8f9fa')
    
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