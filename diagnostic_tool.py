#!/usr/bin/env python3
"""
Diagnostic Tool for Smart Product Pricing Challenge
Identifies and fixes common issues before training
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def check_environment():
    """Check Python environment and dependencies"""
    print("🔍 ENVIRONMENT DIAGNOSTICS")
    print("=" * 50)
    
    # Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check required packages
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm'
    }
    
    optional_packages = {
        'torch': 'PyTorch (for image processing)',
        'PIL': 'Pillow (for image processing)',
        'xgboost': 'XGBoost (for better models)',
        'lightgbm': 'LightGBM (for better models)',
        'nltk': 'NLTK (for text processing)'
    }
    
    missing_required = []
    missing_optional = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - REQUIRED")
            missing_required.append(package)
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError:
            print(f"⚠️  {description} - OPTIONAL")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {missing_optional}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    return True

def check_data_files():
    """Check if data files exist and are valid"""
    print("\n📁 DATA FILE DIAGNOSTICS")
    print("=" * 50)
    
    dataset_folder = 'dataset'
    required_files = ['train.csv', 'test.csv', 'sample_test.csv', 'sample_test_out.csv']
    
    if not os.path.exists(dataset_folder):
        print(f"❌ Dataset folder '{dataset_folder}' not found")
        return False
    
    all_files_ok = True
    
    for filename in required_files:
        filepath = os.path.join(dataset_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ {filename} - NOT FOUND")
            all_files_ok = False
            continue
        
        try:
            # Check file size
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✅ {filename} - {file_size:.1f} MB")
            
            # Quick CSV validation
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath, nrows=5)  # Read first 5 rows
                print(f"   Columns: {list(df.columns)}")
                print(f"   Shape preview: {df.shape[0]}+ rows, {df.shape[1]} columns")
                
                # Check for required columns
                if filename == 'train.csv':
                    required_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        print(f"   ❌ Missing columns: {missing_cols}")
                        all_files_ok = False
                    else:
                        print(f"   ✅ All required columns present")
                
                elif filename == 'test.csv':
                    required_cols = ['sample_id', 'catalog_content', 'image_link']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        print(f"   ❌ Missing columns: {missing_cols}")
                        all_files_ok = False
                    else:
                        print(f"   ✅ All required columns present")
        
        except Exception as e:
            print(f"❌ {filename} - ERROR: {e}")
            all_files_ok = False
    
    return all_files_ok

def check_sample_data():
    """Analyze sample data for potential issues"""
    print("\n📊 SAMPLE DATA ANALYSIS")
    print("=" * 50)
    
    try:
        # Load sample data
        sample_test = pd.read_csv('dataset/sample_test.csv')
        sample_out = pd.read_csv('dataset/sample_test_out.csv')
        
        print(f"Sample test records: {len(sample_test)}")
        print(f"Sample output records: {len(sample_out)}")
        
        # Check for missing values
        missing_content = sample_test['catalog_content'].isna().sum()
        missing_links = sample_test['image_link'].isna().sum()
        
        print(f"Missing catalog_content: {missing_content}")
        print(f"Missing image_link: {missing_links}")
        
        # Check price distribution
        prices = sample_out['price']
        print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"Mean price: ${prices.mean():.2f}")
        print(f"Median price: ${prices.median():.2f}")
        
        # Check for negative prices
        negative_prices = (prices < 0).sum()
        if negative_prices > 0:
            print(f"⚠️  {negative_prices} negative prices found")
        else:
            print("✅ All prices are positive")
        
        # Text length analysis
        text_lengths = sample_test['catalog_content'].str.len()
        print(f"Text length range: {text_lengths.min()} - {text_lengths.max()} characters")
        print(f"Average text length: {text_lengths.mean():.0f} characters")
        
        # Check image URLs
        valid_urls = sample_test['image_link'].str.contains('http', na=False).sum()
        print(f"Valid image URLs: {valid_urls}/{len(sample_test)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data analysis failed: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    print("\n💻 SYSTEM RESOURCES")
    print("=" * 50)
    
    try:
        import psutil
        
        # Memory
        memory = psutil.virtual_memory()
        print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"Memory usage: {memory.percent}%")
        
        if memory.available < 2 * (1024**3):  # Less than 2GB
            print("⚠️  Low memory - consider using quick_solution.py")
        else:
            print("✅ Sufficient memory for full solution")
        
        # CPU
        cpu_count = psutil.cpu_count()
        print(f"CPU cores: {cpu_count}")
        
        # Disk space
        disk = psutil.disk_usage('.')
        print(f"Free disk space: {disk.free / (1024**3):.1f} GB")
        
        if disk.free < 5 * (1024**3):  # Less than 5GB
            print("⚠️  Low disk space - may affect image downloads")
        else:
            print("✅ Sufficient disk space")
        
    except ImportError:
        print("⚠️  psutil not available - install with: pip install psutil")
    except Exception as e:
        print(f"⚠️  Resource check failed: {e}")

def suggest_solutions():
    """Suggest solutions based on diagnostics"""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    print("Based on the diagnostics above:")
    print()
    
    print("🚀 Quick Start Options:")
    print("1. Fast testing: python quick_solution.py")
    print("2. Full solution: python optimized_solution.py")
    print("3. Original solution: python solution.py")
    print()
    
    print("🔧 If you have issues:")
    print("- Low memory: Use quick_solution.py (text-only)")
    print("- Missing packages: pip install -r requirements.txt")
    print("- Image errors: Set use_images=False in the code")
    print("- Slow training: Reduce dataset size for testing")
    print()
    
    print("📊 For better performance:")
    print("- Install optional packages: xgboost, lightgbm")
    print("- Use SSD storage for faster I/O")
    print("- Close other applications to free memory")
    print("- Use optimized_solution.py for best results")

def create_test_script():
    """Create a minimal test script"""
    test_script = '''#!/usr/bin/env python3
"""
Minimal test script to verify setup
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def quick_test():
    """Quick test with sample data"""
    print("🧪 Running quick test...")
    
    try:
        # Load sample data
        sample_test = pd.read_csv('dataset/sample_test.csv')
        sample_out = pd.read_csv('dataset/sample_test_out.csv')
        
        # Simple features
        X = np.array([
            [len(str(text)), len(str(text).split())] 
            for text in sample_test['catalog_content']
        ])
        y = sample_out['price'].values
        
        # Simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Predict
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        
        print(f"✅ Test completed!")
        print(f"MAE: {mae:.2f}")
        print(f"Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    quick_test()
'''
    
    with open('test_setup.py', 'w') as f:
        f.write(test_script)
    
    print("\n🧪 Created test_setup.py - run this to verify your setup")

def main():
    """Run all diagnostics"""
    print("🏥 SMART PRODUCT PRICING - DIAGNOSTIC TOOL")
    print("=" * 60)
    
    all_checks = []
    
    # Run all checks
    all_checks.append(check_environment())
    all_checks.append(check_data_files())
    all_checks.append(check_sample_data())
    check_system_resources()
    
    # Summary
    print("\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if all(all_checks):
        print("🎉 All critical checks passed!")
        print("✅ Ready to run the solution")
    else:
        print("⚠️  Some issues found - see details above")
        print("🔧 Follow the recommendations to fix issues")
    
    suggest_solutions()
    create_test_script()
    
    print("\n" + "=" * 60)
    print("Run 'python test_setup.py' to verify your setup")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''