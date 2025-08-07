#!/usr/bin/env python3
"""
NFL Betting Analytics Pro - Setup Script
Automated setup and configuration for the NFL betting analytics platform
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_config_file():
    """Create configuration file for API keys"""
    config_content = """# NFL Betting Analytics Pro - Configuration File
# Add your API keys here for real-time data integration

# The Odds API (Free tier: 500 requests/month)
# Sign up at: https://the-odds-api.com/
ODDS_API_KEY = "your_odds_api_key_here"

# OpenWeatherMap API (Free tier: 1000 requests/day)
# Sign up at: https://openweathermap.org/api
WEATHER_API_KEY = "your_weather_api_key_here"

# ESPN API (Free, no key required)
# Rate limited but provides good team data
ESPN_API_ENABLED = True

# Mock Data Settings
USE_MOCK_DATA = True  # Set to False when you have real API keys
MOCK_GAMES_PER_WEEK = 16
MOCK_HISTORICAL_YEARS = 5

# Model Settings
MODEL_CONFIDENCE_THRESHOLD = 0.65
MINIMUM_EDGE_THRESHOLD = 1.0  # Minimum edge in points to recommend bet
STRONG_BET_THRESHOLD = 2.5    # Edge threshold for "STRONG BET" recommendations

# Alert Settings
ENABLE_LINE_MOVEMENT_ALERTS = True
LINE_MOVEMENT_THRESHOLD = 1.0  # Alert when line moves this many points
REVERSE_LINE_MOVEMENT_ALERTS = True

# Performance Tracking
TRACK_PERFORMANCE = True
SAVE_PREDICTIONS = True
BACKTEST_ENABLED = True
"""
    
    config_path = Path("config.py")
    if not config_path.exists():
        with open(config_path, "w") as f:
            f.write(config_content)
        print("✅ Configuration file created: config.py")
        print("📝 Edit config.py to add your API keys for real-time data")
    else:
        print("ℹ️  Configuration file already exists")

def create_data_directories():
    """Create necessary data directories"""
    directories = [
        "data",
        "data/historical",
        "data/models",
        "data/predictions",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Data directories created")

def test_installation():
    """Test if the installation is working correctly"""
    print("🧪 Testing installation...")
    
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import sklearn
        print("✅ All core packages imported successfully")
        
        # Test if the main app can be imported
        sys.path.append(os.getcwd())
        from app import NFLBettingAnalytics
        analytics = NFLBettingAnalytics()
        print("✅ Main application can be imported")
        
        # Test AI models
        from ai_models import NFLPredictionModels
        models = NFLPredictionModels()
        print("✅ AI models initialized successfully")
        
        # Test data fetcher
        from data_fetcher import NFLDataFetcher
        fetcher = NFLDataFetcher()
        print("✅ Data fetcher initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def display_startup_info():
    """Display information about how to start the application"""
    print("\n" + "="*60)
    print("🏈 NFL BETTING ANALYTICS PRO - SETUP COMPLETE!")
    print("="*60)
    print("\n🚀 TO START THE APPLICATION:")
    print("   streamlit run app.py")
    print("\n🌐 THEN OPEN YOUR BROWSER TO:")
    print("   http://localhost:8501")
    print("\n📊 FEATURES AVAILABLE:")
    print("   ✅ Live odds analysis (mock data)")
    print("   ✅ 4 AI prediction models")
    print("   ✅ Advanced analytics dashboard")
    print("   ✅ Professional visualizations")
    print("   ✅ Betting recommendations")
    print("\n🔧 FOR REAL-TIME DATA:")
    print("   1. Edit config.py with your API keys")
    print("   2. Set USE_MOCK_DATA = False in config.py")
    print("   3. Restart the application")
    print("\n📚 DOCUMENTATION:")
    print("   See README.md for detailed usage instructions")
    print("\n⚠️  DISCLAIMER:")
    print("   This platform is for educational purposes only.")
    print("   Please gamble responsibly!")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🏈 NFL Betting Analytics Pro - Setup")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Create configuration file
    create_config_file()
    
    # Create data directories
    create_data_directories()
    
    # Test installation
    if not test_installation():
        print("❌ Setup completed but testing failed")
        print("   The application may still work, but please check for errors")
    
    # Display startup information
    display_startup_info()

if __name__ == "__main__":
    main()
