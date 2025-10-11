#!/usr/bin/env python3
"""
Training Monitor - Track progress without interrupting the main training
Run this in a separate terminal while training is running
"""

import os
import time
import psutil
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

class TrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.log_file = "training_progress.log"
        self.image_folder = "images"
        self.dataset_folder = "dataset"
        
    def check_image_progress(self):
        """Check how many images have been downloaded"""
        if not os.path.exists(self.image_folder):
            return 0, "Images folder not created yet"
        
        image_count = len([f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        return image_count, f"{image_count} images downloaded"
    
    def check_model_files(self):
        """Check if model files are being created"""
        model_file = "trained_model.pkl"
        output_file = os.path.join(self.dataset_folder, "test_out.csv")
        
        status = []
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024*1024)  # MB
            status.append(f"✓ Model file: {size:.1f} MB")
        else:
            status.append("⏳ Model file: Not created yet")
            
        if os.path.exists(output_file):
            status.append("✓ Predictions: Completed!")
        else:
            status.append("⏳ Predictions: Not started")
            
        return status
    
    def get_system_stats(self):
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu': cpu_percent,
            'memory_used': memory.used / (1024**3),  # GB
            'memory_total': memory.total / (1024**3),  # GB
            'memory_percent': memory.percent
        }
    
    def estimate_remaining_time(self, current_progress, total_expected):
        """Estimate remaining time based on current progress"""
        if current_progress == 0:
            return "Calculating..."
        
        elapsed = time.time() - self.start_time
        rate = current_progress / elapsed
        remaining_items = total_expected - current_progress
        
        if rate > 0:
            remaining_seconds = remaining_items / rate
            remaining_minutes = remaining_seconds / 60
            return f"{remaining_minutes:.1f} minutes"
        else:
            return "Calculating..."
    
    def display_status(self):
        """Display current training status"""
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        
        print("🔍 TRAINING MONITOR")
        print("=" * 50)
        print(f"⏰ Running for: {(time.time() - self.start_time)/60:.1f} minutes")
        print(f"🕐 Current time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Image download progress
        image_count, image_status = self.check_image_progress()
        print(f"📥 Image Download: {image_status}")
        if image_count > 0:
            # Assuming 75k total images for full dataset
            estimated_total = 75000
            progress_percent = (image_count / estimated_total) * 100
            remaining_time = self.estimate_remaining_time(image_count, estimated_total)
            print(f"    Progress: {progress_percent:.1f}% | ETA: {remaining_time}")
        print()
        
        # Model status
        print("🤖 Model Status:")
        model_status = self.check_model_files()
        for status in model_status:
            print(f"    {status}")
        print()
        
        # System resources
        stats = self.get_system_stats()
        print("💻 System Resources:")
        print(f"    CPU: {stats['cpu']:.1f}%")
        print(f"    Memory: {stats['memory_used']:.1f}/{stats['memory_total']:.1f} GB ({stats['memory_percent']:.1f}%)")
        print()
        
        # Progress indicators
        print("📊 Training Stages:")
        stages = [
            ("Data Loading", "✓" if image_count > 100 else "⏳"),
            ("Feature Extraction", "✓" if image_count > 1000 else "⏳"),
            ("Model Training", "✓" if os.path.exists("trained_model.pkl") else "⏳"),
            ("Predictions", "✓" if os.path.exists(os.path.join(self.dataset_folder, "test_out.csv")) else "⏳")
        ]
        
        for stage, status in stages:
            print(f"    {status} {stage}")
        
        print("\n" + "=" * 50)
        print("Press Ctrl+C to stop monitoring")
        print("This monitor doesn't affect training performance")
    
    def run(self):
        """Run the monitoring loop"""
        print("🚀 Starting Training Monitor...")
        print("This will track your training progress without interrupting it.")
        print("Run your training script in another terminal now!")
        print()
        
        try:
            while True:
                self.display_status()
                time.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped. Training continues in background.")

def main():
    monitor = TrainingMonitor()
    monitor.run()

if __name__ == "__main__":
    main()