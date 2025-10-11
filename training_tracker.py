#!/usr/bin/env python3
"""
Simple training progress tracker
Shows a live progress bar and time estimates
"""

import time
import os
from datetime import datetime

class ProgressTracker:
    def __init__(self, total_steps=6):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        self.log_file = "progress.log"
        
        # Clear previous log
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
    
    def log_step(self, step_name, details=""):
        """Log completion of a step"""
        self.current_step += 1
        current_time = time.time()
        step_duration = current_time - (self.step_times[-1] if self.step_times else self.start_time)
        self.step_times.append(current_time)
        
        # Calculate progress
        progress = (self.current_step / self.total_steps) * 100
        elapsed_total = current_time - self.start_time
        
        # Estimate remaining time
        if self.current_step > 0:
            avg_step_time = elapsed_total / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps * avg_step_time
            eta_minutes = eta_seconds / 60
        else:
            eta_minutes = 0
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Log to file and print
        log_message = f"[{datetime.now().strftime('%H:%M:%S')}] Step {self.current_step}/{self.total_steps}: {step_name}"
        if details:
            log_message += f" - {details}"
        log_message += f" | {step_duration:.1f}s"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
        
        print(f"\n{'='*60}")
        print(f"Progress: [{bar}] {progress:.1f}%")
        print(f"Step {self.current_step}/{self.total_steps}: {step_name}")
        if details:
            print(f"Details: {details}")
        print(f"Step time: {step_duration:.1f}s | Total time: {elapsed_total/60:.1f}min")
        if eta_minutes > 0:
            print(f"ETA: {eta_minutes:.1f} minutes")
        print(f"{'='*60}")
    
    def is_complete(self):
        return self.current_step >= self.total_steps

# Global tracker instance
tracker = ProgressTracker()

def track_progress(step_name, details=""):
    """Convenience function to track progress"""
    tracker.log_step(step_name, details)

def show_final_summary():
    """Show final training summary"""
    total_time = time.time() - tracker.start_time
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average step time: {total_time/tracker.total_steps:.1f} seconds")
    
    with open(tracker.log_file, 'a') as f:
        f.write(f"\nTraining completed at {datetime.now()}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")

if __name__ == "__main__":
    # Test the tracker
    import time
    
    track_progress("Loading Data", "75,000 samples")
    time.sleep(2)
    
    track_progress("Text Processing", "Extracting features")
    time.sleep(3)
    
    track_progress("Image Processing", "ResNet50 features")
    time.sleep(5)
    
    track_progress("Feature Scaling", "StandardScaler")
    time.sleep(1)
    
    track_progress("Model Training", "4 models in ensemble")
    time.sleep(4)
    
    track_progress("Making Predictions", "75,000 predictions")
    time.sleep(2)
    
    show_final_summary()