import pandas as pd
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
import os
from markitdown import MarkItDown
import markdown

app = Flask(__name__)

# Configure Gemini API
# 🔧 CONFIGURATION REQUIRED: Replace with your actual Google AI API key
# Get your API key from: https://aistudio.google.com/app/apikey
genai.configure(api_key="AIzaSyDSPgQBt8uO_HXcMPX434AlEytdvb90i0Q")


# Load the dataset once when the app starts
print("Loading incident data...")
try:
    # 🔧 CONFIGURATION REQUIRED: Update this path to your actual CSV file
    # Place your incident data CSV in the 'data' folder or update the path below
    df = pd.read_csv('data/snow_data.csv')
    print(f"✅ Loaded {len(df)} incident records")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("💡 Please check the CSV file path and format")
    df = None

# Initialize models
keyword_extraction_prompt = """**You are an incident keyword extractor.**
When I supply an incident description, follow these rules:

1. **Extract at most 10 unique keywords or key phrases** that best identify and classify the incident.
2. Treat error codes (e.g., `REF_DATA_ERR_029`), acronyms (e.g., `FDS`, `CA`), system or job names (e.g., `RETLCORR`), and descriptive multi-word clauses that can recur (e.g., `LOC DIV DEPT MDM LOOKUP FAILURE - NO MATCH FOUND`) as candidate keywords.
3. **Preserve the exact casing, punctuation, and order of appearance** from the incident text.
4. Omit generic terms such as "alert", "failure", "error", "incident", "the", etc.
5. Return the result on one line by a comma-separated list (no quotes, no extra text).

**Example**
**Input**
Incident: `<Mainframe Batch Failure AlertCA | FDS | RETLCORR | REF_DATA_ERR_029 | LOC DIV DEPT MDM LOOKUP FAILURE - NO MATCH FOUND`
**Output**
FDS, RETLCORR, REF_DATA_ERR_029, CA, LOC DIV DEPT MDM LOOKUP FAILURE - NO MATCH FOUND"""

#keyword_model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=keyword_extraction_prompt)
keyword_model = genai.GenerativeModel('gemini-2.5-flash')
analysis_model = genai.GenerativeModel('gemini-2.5-pro')

def extract_keywords(user_query):
    """Extract keywords from user query using Gemini"""
    try:
        response = keyword_model.generate_content([
            {"role": "model", "parts": [keyword_extraction_prompt]},
            {"role": "user", "parts": [user_query]}
        ])

        keywords = [keyword.strip() for keyword in response.text.split(',')]
        return keywords
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def find_matching_incidents(keywords, df):
    """Find incidents matching the keywords and score them"""
    if df is None:
        return pd.DataFrame(), 0, 0
    
    # Create a copy to avoid modifying original
    df_work = df.copy()
    
    # Initialize scoring columns
    df_work['keyword_score'] = 0
    df_work['matched_keywords'] = ''
    
    # Search for each keyword
    for keyword in keywords:
        if keyword:
            mask = df_work['All_Details'].str.contains(keyword, case=False, na=False)
            df_work.loc[mask, 'keyword_score'] += 1
            df_work.loc[mask, 'matched_keywords'] = df_work.loc[mask, 'matched_keywords'].apply(
                lambda x: f"{x}, {keyword}" if x else keyword
            )
    
    # Get top scoring incidents
    df_sorted = df_work.sort_values('keyword_score', ascending=False)
    matched_rows = df_sorted[df_sorted['keyword_score'] > 0]
    
    # Store total count of matched incidents
    total_matches = len(matched_rows)
    
    # Count incidents with score >= 80% of keyword count (floor side)
    high_score_threshold = max(1, int(len(keywords) * 0.8))
    high_score_matches = len(matched_rows[matched_rows['keyword_score'] >= high_score_threshold])
    
    # Get top 10 or all if less than 10
    num_top_rows = min(10, len(matched_rows))
    top_results = matched_rows.head(num_top_rows)
    
    return top_results[['Number', 'All_Details', 'keyword_score', 'matched_keywords']], total_matches, high_score_matches

def format_for_analysis(user_query, keywords, top_incidents):
    """Format the data for LLM analysis"""
    markdown_output = "# Top Scoring Incident Records for LLM Analysis\n\n"
    markdown_output += f"**Search Query:** `{user_query}`\n\n"
    markdown_output += f"**Extracted Keywords:** {', '.join([f'`{k}`' for k in keywords])}\n\n"
    markdown_output += f"**Total Records Found:** {len(top_incidents)}\n\n"
    markdown_output += "---\n\n"
    
    for idx, (_, row) in enumerate(top_incidents.iterrows(), 1):
        markdown_output += f"## Incident Record #{idx}\n\n"
        markdown_output += f"**Number:** `{row['Number']}`\n\n"
        markdown_output += f"**Score:** {row['keyword_score']}/5 keywords matched\n\n"
        markdown_output += f"**Matched Keywords:** {row['matched_keywords']}\n\n"
        markdown_output += f"**All Details:**\n```\n{row['All_Details']}\n```\n\n"
        markdown_output += "---\n\n"
    
    return markdown_output

def analyze_incidents(user_query, keywords, formatted_data):
    """Get insights from Gemini Pro"""
    analysis_prompt = f"""
Act as a Senior IT Operations Analyst. You have been given the following raw incident data. Your goal is to analyze it and present a clear, actionable report for management.

**Incident Data:**
----------------
{formatted_data}
----------------

**Current High-Priority Incident:** "{user_query}"

**Your report should be structured as follows:**

**Part 1: Executive Summary**
*   A brief overview of the key findings and most critical recommendations.

**Part 2: Trend & Pattern Analysis 📈**
*   Who are the people usually involved in these incidents?
*   What are the most common categories of incidents?
*   Are there any correlations? (e.g., do certain events trigger other incidents?)
*   Visualize or describe any notable trends in the data.

**Part 3: Deep Dive - Root Cause Analysis 🧐**
*   **What Happened:** For each major incident pattern, describe the sequence of events.
*   **Why It Happened:** Analyze the underlying factors (e.g., software bugs, user error, infrastructure failure) that contributed to the incidents.

**Part 4: Strategic Recommendations for Prevention ✅**
*   **How to Prevent It:** Propose a list of concrete preventative measures.
*   Categorize recommendations by effort (Low, Medium, High) and impact (Low, Medium, High).

**Part 5: Action Plan for the Current Incident ("{user_query}") 📢**
*   **Immediate Steps:** What should be done *right now* to mitigate the impact of this incident?
*   **Follow-Up Actions:** What are the next steps for full resolution and post-incident review?

**Final Instructions:**
*   The analysis must be strictly based on the provided data.
*   Use emojis to enhance readability.
*   The tone should be professional and analytical.
*   After each part, add a line break for clarity. Also add 2 blank line before the next part.
*   Make the whole information very crisp and clear and avoid repeatative information. 
"""
    
    try:
        response = analysis_model.generate_content(analysis_prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing incidents: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        user_query = request.json.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Please enter an incident description'})
        
        if df is None:
            return jsonify({'error': 'Dataset not loaded. Please check the data file path.'})
        
        # Step 1: Extract keywords
        print(f"Extracting keywords for: {user_query}")
        keywords = extract_keywords(user_query)
        
        if not keywords:
            return jsonify({'error': 'Could not extract keywords from the query'})
        
        # Step 2: Find matching incidents
        print(f"Found keywords: {keywords}")
        top_incidents, total_matches, high_score_matches = find_matching_incidents(keywords, df)
        
        if total_matches == 0:
            return jsonify({
                'query': user_query,
                'keywords': keywords,
                'matches_found': 0,
                'high_score_matches': 0,
                'analysis': 'No matching incidents found in the database for the extracted keywords.'
            })
        
        # Step 3: Format for analysis
        formatted_data = format_for_analysis(user_query, keywords, top_incidents)
        
        # Step 4: Get AI analysis
        print(f"Analyzing {len(top_incidents)} matching incidents...")
        analysis_result = analyze_incidents(user_query, keywords, formatted_data)
        
        # Convert markdown analysis to HTML
        analysis_html = markdown.markdown(analysis_result)
        
        # Step 5: Prepare response
        incident_summary = []
        for idx, (_, row) in enumerate(top_incidents.iterrows(), 1):
            incident_summary.append({
                'rank': idx,
                'number': row['Number'],
                'score': f"{row['keyword_score']}/{len(keywords)}",
                'matched_keywords': row['matched_keywords'],
                'details': row['All_Details']
            })
        
        return jsonify({
            'query': user_query,
            'keywords': keywords,
            'matches_found': total_matches,
            'high_score_matches': high_score_matches,
            'incidents': incident_summary,
            'analysis': analysis_html,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'})
#comments
if __name__ == '__main__':
    print("🚀 Starting Incident Analyzer App...")
    print(f"📊 Dataset status: {'✅ Loaded' if df is not None else '❌ Not loaded'}")
    print("🌐 Access the app at: http://localhost:5050")
    app.run(debug=True, host='0.0.0.0', port=5050)
