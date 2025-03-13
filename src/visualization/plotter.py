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
    
    def __init__(self, dark_mode=False):
        """Initialize the PerformanceVisualizer
        
        Args:
            dark_mode (bool): Whether to use dark theme for plots
        """
        # Set up style based on mode
        self.dark_mode = dark_mode
        if dark_mode:
            plt.style.use('dark_background')
            self.bg_color = '#2d2d2d'
            self.text_color = 'white'
            self.grid_color = '#404040'
            self.metrics_box_color = '#404040'
        else:
            sns.set_theme(style="whitegrid")
            self.bg_color = '#f8f9fa'
            self.text_color = 'black'
            self.grid_color = '#e0e0e0'
            self.metrics_box_color = 'lightgray'
        
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
        
        # Determine lifting phases using force thresholds
        mean_force = np.mean(forces) if len(forces) > 0 else 0
        lifting_mask = forces > mean_force
        lowering_mask = forces <= mean_force
        
        # Plot lifting phase (higher force)
        if np.any(lifting_mask):
            ax.fill_between(times, forces, mean_force, where=lifting_mask, 
                          color=self.color_palette['force'], alpha=0.3,
                          label='Lifting Phase (↑)')
        
        # Plot lowering phase (lower force)
        if np.any(lowering_mask):
            ax.fill_between(times, forces, mean_force, where=lowering_mask, 
                          color=self.color_palette['force'], alpha=0.15,
                          label='Lowering Phase (↓)')
        
        # Plot the force line
        ax.plot(times, forces, color=self.color_palette['force'], 
               linewidth=1.5, label='Applied Force')
        
        # Add mean line
        if len(forces) > 0:
            ax.axhline(y=mean_force, color=self.color_palette['force'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            ax.text(times[-1], mean_force, f' {mean_force:.1f} N', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        # Add legend with phase information
        ax.legend(loc='best', fontsize=8, frameon=True, fancybox=True)
        
        ax.set_xlabel('Time (s)', fontsize='small')
        ax.set_ylabel('Force (N)', fontsize='small')
        ax.set_title('Applied Force', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize='small')
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_facecolor('#f8f9fa')
    
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
        
        # Highlight positive power (concentric phase) - most important for performance
        pos_mask = powers > 0
        if np.any(pos_mask):
            ax.fill_between(times, powers, 0, where=pos_mask, 
                          color=self.color_palette['power'], alpha=0.3, 
                          label='Concentric Power')
        
        # Plot the power line
        ax.plot(times, powers, color=self.color_palette['power'], 
               linewidth=1.5, label='Power Output')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=0.8)
        
        # Calculate and show mean for positive power
        if np.any(pos_mask):
            mean_power = np.mean(powers[pos_mask])
            ax.axhline(y=mean_power, color=self.color_palette['power'], 
                     linestyle='--', alpha=0.7, linewidth=1,
                     label=f'Mean Power: {mean_power:.1f} W')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Output Over Time')
        
        # Add a nicer legend with smaller font
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
        
        # Add denser grid with both major and minor lines
        ax.grid(True, which='major', alpha=0.4, linestyle='-', color=self.grid_color)
        ax.grid(True, which='minor', alpha=0.2, linestyle=':', color=self.grid_color)
        ax.minorticks_on()  # Enable minor ticks
        
        # Set more granular y-axis ticks
        if len(powers) > 0:
            max_power = np.max(np.abs(powers))
            # Calculate a nice round step size that gives us ~10-20 major ticks
            step = 10 ** np.floor(np.log10(max_power / 10))  # Start with order of magnitude
            if max_power / step > 20:
                step *= 2
            elif max_power / step < 10:
                step /= 2
            
            major_ticks = np.arange(0, max_power + step, step)
            minor_ticks = np.arange(0, max_power + step/2, step/2)
            ax.yaxis.set_major_locator(plt.FixedLocator(major_ticks))
            ax.yaxis.set_minor_locator(plt.FixedLocator(minor_ticks))
        
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self, time_series_data: Dict[str, pd.DataFrame], 
                               fig_width=10, fig_height=8) -> Figure:
        """
        Create a comprehensive dashboard with key performance graphs.
        
        Args:
            time_series_data (Dict[str, pd.DataFrame]): Dictionary of time series data
            fig_width (float): Width of the figure in inches
            fig_height (float): Height of the figure in inches
           
        Returns:
            Figure: Matplotlib figure with the dashboard
        """
        # Create figure with a more efficient layout
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create a grid with better space utilization
        gs = fig.add_gridspec(3, 2, height_ratios=[0.15, 1, 1], 
                            hspace=0.4, wspace=0.3,
                            left=0.08, right=0.92, 
                            bottom=0.08, top=0.92)
        
        fig.suptitle('Weightlifting Performance Analysis', 
                    fontsize=12, fontweight='bold', y=0.99)
        
        # Calculate metrics from time series data
        metrics = {}
        
        # Rep count from velocity data
        rep_count = 0
        if 'velocities' in time_series_data:
            velocity_data = time_series_data['velocities']
            if 'bar_velocity' in velocity_data.columns:
                velocities = velocity_data['bar_velocity'].values
                # Simple rep counting based on zero crossings
                pos_to_neg = 0
                for i in range(1, len(velocities)):
                    if velocities[i-1] >= 0 and velocities[i] < 0:
                        pos_to_neg += 1
                rep_count = pos_to_neg
                
                # Calculate velocity metrics
                pos_mask = velocities > 0
                neg_mask = velocities < 0
                
                if np.any(pos_mask):
                    metrics['avg_velocity'] = np.mean(velocities[pos_mask])
                    metrics['max_velocity'] = np.max(velocities[pos_mask])
                else:
                    metrics['avg_velocity'] = 0
                    metrics['max_velocity'] = 0
        
        # Force metrics
        if 'forces' in time_series_data:
            force_data = time_series_data['forces']
            if 'bar_force' in force_data.columns:
                forces = force_data['bar_force'].values
                if len(forces) > 0:
                    metrics['avg_force'] = np.mean(forces)
                    metrics['max_force'] = np.max(forces)
                else:
                    metrics['avg_force'] = 0
                    metrics['max_force'] = 0
        
        # Power metrics
        if 'powers' in time_series_data:
            power_data = time_series_data['powers']
            if 'bar_power' in power_data.columns:
                powers = power_data['bar_power'].values
                pos_mask = powers > 0
                if np.any(pos_mask):
                    metrics['avg_power'] = np.mean(powers[pos_mask])
                    metrics['max_power'] = np.max(powers[pos_mask])
                else:
                    metrics['avg_power'] = 0
                    metrics['max_power'] = 0
        
        # Create a row for metrics at the top spanning both columns
        metrics_ax = fig.add_subplot(gs[0, :])
        metrics_ax.axis('off')
        
        # Format metrics text with improved spacing
        metrics_boxes = [
            {'text': f"Repetitions\n{rep_count}", 'x': 0.125},
            {'text': f"Velocity\nMax: {metrics.get('max_velocity', 0):.2f} m/s\nAvg: {metrics.get('avg_velocity', 0):.2f} m/s", 'x': 0.375},
            {'text': f"Force\nMax: {metrics.get('max_force', 0):.1f} N\nAvg: {metrics.get('avg_force', 0):.1f} N", 'x': 0.625},
            {'text': f"Power\nMax: {metrics.get('max_power', 0):.1f} W\nAvg: {metrics.get('avg_power', 0):.1f} W", 'x': 0.875}
        ]
        
        for box in metrics_boxes:
            metrics_ax.text(box['x'], 0.5, box['text'],
                          ha='center', va='center', fontsize=10, fontweight='bold',
                          color=self.text_color,
                          bbox=dict(facecolor=self.metrics_box_color, 
                                  alpha=0.5, 
                                  boxstyle='round,pad=0.6',
                                  edgecolor=self.grid_color))

        # Add vertical separators
        for x in [0.25, 0.50, 0.75]:
            metrics_ax.axvline(x=x, color=self.grid_color, 
                             linestyle=':', alpha=0.5, linewidth=1)

        # Create subplots with maximized space
        if 'angles' in time_series_data and not time_series_data['angles'].empty:
            ax_angles = fig.add_subplot(gs[1, 0])
            self._plot_joint_angles_subplot(ax_angles, time_series_data['angles'])
            ax_angles.set_position(ax_angles.get_position().expanded(1.1, 1.1))
        
        if 'velocities' in time_series_data and not time_series_data['velocities'].empty:
            ax_velocity = fig.add_subplot(gs[1, 1])
            self._plot_velocity_subplot_enhanced(ax_velocity, time_series_data['velocities'])
            ax_velocity.set_position(ax_velocity.get_position().expanded(1.1, 1.1))
        
        if 'forces' in time_series_data and not time_series_data['forces'].empty:
            ax_force = fig.add_subplot(gs[2, 0])
            self._plot_force_subplot_enhanced(ax_force, time_series_data['forces'])
            ax_force.set_position(ax_force.get_position().expanded(1.1, 1.1))
            
        if 'powers' in time_series_data and not time_series_data['powers'].empty:
            ax_power = fig.add_subplot(gs[2, 1])
            self._plot_power_subplot_enhanced(ax_power, time_series_data['powers'])
            ax_power.set_position(ax_power.get_position().expanded(1.1, 1.1))
        
        return fig
    
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
            ax.text(times[-1], pos_mean, f'd {pos_mean:.2f}', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        if np.any(neg_mask):
            neg_mean = np.mean(velocities[neg_mask])
            ax.axhline(y=neg_mean, color=self.color_palette['velocity_neg'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            ax.text(times[-1], neg_mean, f'n {neg_mean:.2f}', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        # Add legend with phase information
        legend_elements = [
            plt.Line2D([0], [0], color=self.color_palette['velocity'], lw=1.5, label='Bar Velocity'),
            plt.Rectangle((0, 0), 1, 1, fc=self.color_palette['velocity_pos'], alpha=0.3, label='Concentric (↑)'),
            plt.Rectangle((0, 0), 1, 1, fc=self.color_palette['velocity_neg'], alpha=0.3, label='Eccentric (↓)'),
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
        
        # Determine lifting phases using force thresholds
        mean_force = np.mean(forces) if len(forces) > 0 else 0
        lifting_mask = forces > mean_force
        lowering_mask = forces <= mean_force
        
        # Plot lifting phase (higher force)
        if np.any(lifting_mask):
            ax.fill_between(times, forces, mean_force, where=lifting_mask, 
                          color=self.color_palette['force'], alpha=0.3,
                          label='Lifting Phase (↑)')
        
        # Plot lowering phase (lower force)
        if np.any(lowering_mask):
            ax.fill_between(times, forces, mean_force, where=lowering_mask, 
                          color=self.color_palette['force'], alpha=0.15,
                          label='Lowering Phase (↓)')
        
        # Plot the force line
        ax.plot(times, forces, color=self.color_palette['force'], 
               linewidth=1.5, label='Applied Force')
        
        # Add mean line
        if len(forces) > 0:
            ax.axhline(y=mean_force, color=self.color_palette['force'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            ax.text(times[-1], mean_force, f' {mean_force:.1f} N', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        # Add legend with phase information
        ax.legend(loc='best', fontsize=8, frameon=True, fancybox=True)
        
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
        
        # Highlight positive power (concentric phase)
        pos_mask = powers > 0
        neg_mask = powers <= 0
        
        # Plot concentric phase (positive power)
        if np.any(pos_mask):
            ax.fill_between(times, powers, 0, where=pos_mask, 
                          color=self.color_palette['power'], alpha=0.3,
                          label='Concentric Phase (↑)')
            # Calculate mean only for concentric phase
            pos_mean = np.mean(powers[pos_mask])
        else:
            pos_mean = 0
        
        # Plot eccentric phase (negative power)
        if np.any(neg_mask):
            ax.fill_between(times, powers, 0, where=neg_mask, 
                          color=self.color_palette['power'], alpha=0.15,
                          label='Eccentric Phase (↓)')
        
        # Plot the power line
        ax.plot(times, powers, color=self.color_palette['power'], linewidth=1.5, 
               label='Power Output')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=0.8)
        
        # Add mean line for concentric phase only
        if pos_mean > 0:
            ax.axhline(y=pos_mean, color=self.color_palette['power'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            ax.text(times[-1], pos_mean, f' {pos_mean:.1f} W', va='center', fontsize=8,
                   backgroundcolor='white', alpha=0.7)
        
        # Add legend with phase information
        ax.legend(loc='best', fontsize=8, frameon=True, fancybox=True)
        
        ax.set_xlabel('Time (s)', fontsize='small')
        ax.set_ylabel('Power (W)', fontsize='small')
        ax.set_title('Power Output', fontsize=10, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize='small')
        
        # Add denser grid with both major and minor lines
        ax.grid(True, which='major', alpha=0.4, linestyle='-', color=self.grid_color)
        ax.grid(True, which='minor', alpha=0.2, linestyle=':', color=self.grid_color)
        ax.minorticks_on()  # Enable minor ticks
        
        # Set more granular y-axis ticks
        if len(powers) > 0:
            max_power = np.max(np.abs(powers))
            # Calculate a nice round step size that gives us ~10-20 major ticks
            step = 10 ** np.floor(np.log10(max_power / 10))  # Start with order of magnitude
            if max_power / step > 20:
                step *= 2
            elif max_power / step < 10:
                step /= 2
            
            major_ticks = np.arange(0, max_power + step, step)
            minor_ticks = np.arange(0, max_power + step/2, step/2)
            ax.yaxis.set_major_locator(plt.FixedLocator(major_ticks))
            ax.yaxis.set_minor_locator(plt.FixedLocator(minor_ticks))
        
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