#!/usr/bin/env python3
"""
Fast Setup for 10K Training
"""

import os
import subprocess
import sys

def install_minimal_requirements():
    """Install only essential packages"""
    print("📦 Installing minimal requirements...")
    
    essential_packages = [
        'pandas>=1.5.0',
        'numpy>=1.21.0', 
        'scikit-learn>=1.1.0',
        'tqdm>=4.64.0'
    ]
    
    for package in essential_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL)
            print(f"✅ {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    # Try to install XGBoost (optional)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "xgboost"
        ], stdout=subprocess.DEVNULL)
        print("✅ xgboost (optional)")
    except:
        print("⚠️  xgboost (optional) - install manually if needed")

def check_data():
    """Check if data files exist"""
    print("\n📁 Checking data files...")
    
    if not os.path.exists('dataset'):
        print("❌ dataset/ folder not found")
        print("   Please create dataset/ folder and add train.csv, test.csv")
        return False
    
    required_files = ['train.csv', 'test.csv']
    for file in required_files:
        path = os.path.join('dataset', file)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} not found")
            return False
    
    return True

def main():
    print("🚀 FAST SETUP FOR 10K TRAINING")
    print("=" * 40)
    
    # Install packages
    install_minimal_requirements()
    
    # Check data
    data_ok = check_data()
    
    print("\n" + "=" * 40)
    if data_ok:
        print("✅ Setup complete! Ready to train.")
        print("\nRun: python train_10k.py")
    else:
        print("⚠️  Please add dataset files first.")
    print("=" * 40)

if __name__ == "__main__":
    main()