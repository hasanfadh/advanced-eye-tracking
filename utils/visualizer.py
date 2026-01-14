import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os

class EyeTrackingVisualizer:
    """
    Creates visualizations for eye tracking data
    """
    
    def __init__(self, output_dir='data/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_gaze_trajectory(self, data_history, save=True):
        """Plot gaze trajectory over time"""
        df = pd.DataFrame(data_history)
        df_detected = df[df['face_detected'] == True].copy()
        
        if len(df_detected) == 0:
            print("⚠️ No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Horizontal gaze
        ax1.plot(df_detected.index, df_detected['gaze_h_ratio'], 
                color='blue', linewidth=1, alpha=0.7)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.15, color='red', linestyle='--', alpha=0.3, label='Threshold')
        ax1.axhline(y=-0.15, color='red', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Horizontal Gaze (-1=Left, +1=Right)', fontsize=12)
        ax1.set_title('Gaze Trajectory Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Vertical gaze
        ax2.plot(df_detected.index, df_detected['gaze_v_ratio'], 
                color='green', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.15, color='red', linestyle='--', alpha=0.3, label='Threshold')
        ax2.axhline(y=-0.15, color='red', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Frame Number', fontsize=12)
        ax2.set_ylabel('Vertical Gaze (-1=Up, +1=Down)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'gaze_trajectory_{timestamp}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to: {filepath}")
        
        plt.show()
    
    def plot_gaze_heatmap(self, data_history, bins=20, save=True):
        """Create 2D heatmap of gaze positions"""
        df = pd.DataFrame(data_history)
        df_detected = df[df['face_detected'] == True].copy()
        
        if len(df_detected) == 0:
            print("⚠️ No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(
            df_detected['gaze_h_ratio'],
            df_detected['gaze_v_ratio'],
            bins=bins,
            range=[[-1, 1], [-1, 1]]
        )
        
        # Plot heatmap
        im = ax.imshow(h.T, origin='lower', extent=[-1, 1, -1, 1],
                      cmap='hot', interpolation='gaussian', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fixation Count', fontsize=12)
        
        # Add grid and labels
        ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=0, color='cyan', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel('Horizontal Gaze (Left ← → Right)', fontsize=12)
        ax.set_ylabel('Vertical Gaze (Up ← → Down)', fontsize=12)
        ax.set_title('Gaze Position Heatmap', fontsize=14, fontweight='bold')
        
        # Add direction labels
        ax.text(0, 0.9, 'DOWN', ha='center', va='center', fontsize=10, 
               color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax.text(0, -0.9, 'UP', ha='center', va='center', fontsize=10, 
               color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax.text(0.9, 0, 'RIGHT', ha='center', va='center', fontsize=10, 
               color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax.text(-0.9, 0, 'LEFT', ha='center', va='center', fontsize=10, 
               color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'gaze_heatmap_{timestamp}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {filepath}")
        
        plt.show()
    
    def plot_gaze_distribution(self, data_history, save=True):
        """Plot distribution of gaze directions"""
        df = pd.DataFrame(data_history)
        df_detected = df[df['face_detected'] == True].copy()
        
        if len(df_detected) == 0:
            print("⚠️ No data to plot")
            return
        
        gaze_counts = df_detected['gaze_direction'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(gaze_counts)))
        bars = ax.bar(range(len(gaze_counts)), gaze_counts.values, color=colors)
        
        ax.set_xticks(range(len(gaze_counts)))
        ax.set_xticklabels(gaze_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Frame Count', fontsize=12)
        ax.set_title('Gaze Direction Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        total = gaze_counts.sum()
        for i, (bar, count) in enumerate(zip(bars, gaze_counts.values)):
            height = bar.get_height()
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{percentage:.1f}%',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'gaze_distribution_{timestamp}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to: {filepath}")
        
        plt.show()
    
    def plot_blink_analysis(self, data_history, save=True):
        """Plot blink events and EAR over time"""
        df = pd.DataFrame(data_history)
        df_detected = df[df['face_detected'] == True].copy()
        
        if len(df_detected) == 0:
            print("⚠️ No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # EAR over time
        avg_ear = (df_detected['left_ear'] + df_detected['right_ear']) / 2
        ax1.plot(df_detected.index, avg_ear, color='blue', linewidth=1, alpha=0.7)
        ax1.axhline(y=0.21, color='red', linestyle='--', linewidth=2, 
                   alpha=0.5, label='Blink Threshold')
        
        # Mark blink events
        blink_frames = df_detected[df_detected['blink_detected'] == True]
        if len(blink_frames) > 0:
            ax1.scatter(blink_frames.index, 
                       (blink_frames['left_ear'] + blink_frames['right_ear']) / 2,
                       color='red', s=100, marker='x', linewidths=2, 
                       label='Blink Detected', zorder=5)
        
        ax1.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=12)
        ax1.set_title('Blink Detection Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Cumulative blinks
        ax2.plot(df_detected.index, df_detected['blink_count'], 
                color='green', linewidth=2)
        ax2.set_xlabel('Frame Number', fontsize=12)
        ax2.set_ylabel('Cumulative Blink Count', fontsize=12)
        ax2.set_title('Cumulative Blinks Over Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'blink_analysis_{timestamp}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Blink analysis saved to: {filepath}")
        
        plt.show()
    
    def create_comprehensive_report(self, data_history, save=True):
        """Create comprehensive visualization report"""
        print("\nGenerating comprehensive visualization report...")
        
        self.plot_gaze_trajectory(data_history, save=save)
        self.plot_gaze_heatmap(data_history, save=save)
        self.plot_gaze_distribution(data_history, save=save)
        self.plot_blink_analysis(data_history, save=save)
        
        print("\nAll visualizations generated!")