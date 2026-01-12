# 🚀 Incident Analyzer App

An AI-powered incident analysis tool that helps identify patterns and provides insights from historical incident data.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-red?style=for-the-badge&logo=flask)
![AI](https://img.shields.io/badge/AI-Gemini%202.0-purple?style=for-the-badge)

## 🌟 Features

### 🔍 **Smart Analysis**
- **AI-Powered Keyword Extraction**: Uses Google's Gemini 2.0 Flash for intelligent keyword identification
- **Pattern Recognition**: Analyzes 1,365+ historical incidents to find matching patterns
- **Similarity Scoring**: Advanced scoring algorithm with 80% threshold for high-confidence matches

### 🎨 **Modern Dashboard**
- **Walmart-Inspired Design**: Professional color scheme with futuristic elements
- **Google-Style Search**: Familiar search interface with 60% width, center-aligned
- **Real-Time Ticker**: Auto-scrolling incident feed showing high-score matches
- **Responsive Layout**: 3-column grid layout that adapts to different screen sizes

### 📊 **Interactive Features**
- **Live Progress Tracking**: 7-step analysis progress with animated indicators
- **Dynamic Results**: Real-time display of analysis summary, keywords, and insights
- **Detailed Incident Cards**: Expandable cards with full incident details
- **Executive Reports**: AI-generated insights with actionable recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google AI API Key (Gemini)
- CSV incident data file

### Installation

1. **Clone/Download** this folder to your local machine

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   - Open `incident_analyzer_app.py`
   - Replace `"YOUR_API_KEY_HERE"` with your Google AI API key:
   ```python
   genai.configure(api_key="your-actual-api-key-here")
   ```

4. **Prepare Data File**
   - Place your incident CSV file in the appropriate location
   - Update the file path in `incident_analyzer_app.py`:
   ```python
   df = pd.read_csv('/path/to/your/incident_data.csv')
   ```

5. **Add Hero Image** (Optional)
   - Place `herosection.png` in the `static/` folder for custom header background
   - Or remove the background-image CSS property to use gradient background

### Running the App

```bash
python incident_analyzer_app.py
```

Access the application at: **http://localhost:5000**

## 📋 File Structure

```
incident-analyzer-app/
├── incident_analyzer_app.py    # Main Flask application
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── templates/
│   └── index.html              # Main dashboard template
├── static/
│   └── herosection.png         # Header background image (optional)
└── data/
    └── (place your CSV file here)
```

## 🔧 Configuration

### API Configuration
The app uses Google's Gemini 2.0 Flash model for AI analysis. You'll need:
- A Google AI Studio account
- Generated API key
- Update the API key in the main Python file

### Data Requirements
Your CSV file should contain:
- `Number`: Incident identifier
- `All_Details`: Full incident description text
- Other columns as needed

### Customization Options
- **Colors**: Modify CSS variables for different color schemes
- **Ticker Speed**: Adjust animation duration in CSS (currently 30s)
- **Scoring Threshold**: Change the 80% threshold in `find_matching_incidents()`
- **Progress Steps**: Customize the 7-step analysis workflow

## 🎯 How It Works

### 1. **Keyword Extraction**
Uses Gemini AI to extract up to 10 key technical terms, error codes, and system names from incident descriptions.

### 2. **Pattern Matching**
Searches through historical incident database using case-insensitive matching to find similar issues.

### 3. **Intelligent Scoring**
Calculates similarity scores based on keyword matches with an 80% threshold for high-confidence results.

### 4. **AI Analysis**
Generates comprehensive insights including:
- Executive summary
- Trend analysis
- Root cause analysis
- Prevention recommendations
- Immediate action plans

### 5. **Dashboard Display**
Presents results in an intuitive dashboard with:
- Real-time statistics
- Interactive keyword tags
- Auto-scrolling incident ticker
- Detailed analysis reports

## 🛠 Technical Stack

- **Backend**: Flask (Python web framework)
- **AI Engine**: Google Gemini 2.0 Flash
- **Data Processing**: Pandas
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Styling**: Custom CSS with gradients and animations
- **Markdown**: For AI-generated reports

## 📈 Performance Features

- **Fast Keyword Extraction**: ~1-2 seconds using Gemini API
- **Efficient Data Processing**: Optimized pandas operations
- **Real-time Updates**: Asynchronous JavaScript for smooth UX
- **Memory Efficient**: CSV loaded once at startup
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🔒 Security Notes

- Keep your API key secure and never commit it to version control
- Consider using environment variables for production deployment
- Validate input data to prevent injection attacks
- Use HTTPS in production environments

## 🚀 Deployment Options

### Local Development
```bash
python incident_analyzer_app.py
```

### Production Deployment
- Use WSGI server like Gunicorn
- Set up reverse proxy with Nginx
- Configure environment variables
- Enable HTTPS with SSL certificates

## 📞 Support & Documentation

For issues or questions:
1. Check the console output for error messages
2. Verify API key and data file paths
3. Ensure all dependencies are installed
4. Review the CSV file format requirements

## 🎨 UI Components

- **Header**: Hero image background with overlay text
- **Search Bar**: Google-style input with rounded corners
- **Progress Tracker**: 7-step animated progress indicator
- **Results Grid**: 3-column responsive layout
- **Incident Ticker**: Auto-scrolling news-style feed
- **Analysis Cards**: Expandable sections with detailed insights

## 📊 Data Visualization

The dashboard includes:
- Statistical summary cards
- Interactive keyword clouds
- Real-time incident counters
- Trend analysis charts (in AI reports)
- Progress indicators

---

**Built with ❤️ using Flask, Gemini AI, and modern web technologies**

*Ready for production deployment and enterprise use*
