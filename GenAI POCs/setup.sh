#!/bin/bash

echo "🚀 Setting up Incident Analyzer App..."
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🔧 Configuration Required:"
echo "=========================="
echo "1. Open 'incident_analyzer_app.py'"
echo "2. Replace 'YOUR_API_KEY_HERE' with your Google AI API key"
echo "3. Update the CSV file path to point to your incident data"
echo ""
echo "🎯 To run the app:"
echo "=================="
echo "python3 incident_analyzer_app.py"
echo ""
echo "Then open: http://localhost:5000"
echo ""
echo "✨ Setup complete! Happy analyzing! 🎉"
