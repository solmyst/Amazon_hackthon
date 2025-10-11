#!/usr/bin/env python3
"""
Setup script for Smart Product Pricing Challenge
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def check_data_files():
    """Check if dataset files exist"""
    print("Checking dataset files...")
    required_files = [
        'dataset/train.csv',
        'dataset/test.csv',
        'dataset/sample_test.csv',
        'dataset/sample_test_out.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ Found {file_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All dataset files found!")
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    directories = ['images', 'models', 'outputs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created {directory}/ directory")
        else:
            print(f"✓ {directory}/ directory already exists")

def main():
    """Main setup function"""
    print("=== Smart Product Pricing Challenge Setup ===\n")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7 or higher is required")
        return
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create directories
    create_directories()
    
    # Check data files
    if not check_data_files():
        print("\n⚠️  Some dataset files are missing. Please ensure all CSV files are in the dataset/ folder.")
    
    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Run 'python experiment.ipynb' to explore the data")
    print("2. Run 'python quick_solution.py' for a fast text-only solution")
    print("3. Run 'python solution.py' for the full multimodal solution")
    print("\nGood luck with the challenge! 🚀")

if __name__ == "__main__":
    main()