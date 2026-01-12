from flask import Flask, render_template_string, request
import pandas as pd
import os
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from collections import defaultdict
import re

app = Flask(__name__)

# Set your Google API Key
# It's highly recommended to load this from an environment variable for security:
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# For local testing, you can place it directly here, but REMEMBER TO SECURE IT IN PRODUCTION!
GOOGLE_API_KEY = '***'  # <<< REPLACE THIS WITH YOUR ACTUAL GEMINI API KEY!
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# --- DEBUGGING TIP: UNCOMMENT THE SECTION BELOW TO LIST AVAILABLE MODELS ---
# print("Checking available Gemini models...")
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(f"- Generate Content Model: {m.name} (Description: {m.description})")
#     if 'embedContent' in m.supported_generation_methods:
#         print(f"- Embed Content Model: {m.name} (Description: {m.description})")
# print("--- End of Model List ---")
# --- END DEBUGGING TIP ---

# Load Dataset
# !! IMPORTANT: Update this path to where your incident.xlsx file is located.
dataset_path = 'C:\\Users\\s0s0cqc\\Downloads\\incident.xlsx'
try:
    df = pd.read_excel(dataset_path, engine='openpyxl')
    df = df.dropna(axis=1, how='all')
    print(f"Dataset loaded successfully with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: Dataset not found at {dataset_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()


# --- Keyword Generation Logic (NEW/MODIFIED) ---
def generate_comprehensive_keywords(dataframe):
    all_keywords = defaultdict(int)

    # Expanded list of stop words, including common incident-related terms
    stop_words = set([
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'for', 'with',
        'and', 'or', 'but', 'not', 'to', 'of', 'from', 'by', 'this', 'that', 'it',
        'he', 'she', 'we', 'you', 'they', 'i', 'my', 'me', 'our', 'us', 'your',
        'their', 'its', 'has', 'have', 'had', 'do', 'does', 'did', 'can', 'will',
        'would', 'should', 'could', 'get', 'getting', 'tried', 'shows', 'might',
        'when', 'where', 'how', 'what', 'who', 'which', 'whom', 'be', 'been', 'being',
        'as', 'then', 'than', 'more', 'less', 'most', 'some', 'any', 'no', 'only',
        'also', 'very', 'just', 'much', 'many', 'said', 'says', 'user', 'users',
        'issue', 'problem', 'error', 'message', 'cannot', 'cant', 'not working',
        'failed', 'failure', 'unable', 'unresponsive', 'disconnects', 'crashing',
        'offline', 'access', 'connection', 'files', 'documents', 'login', 'site',
        'department', 'printer', 'emails', 'large', 'attachments', 'frequently',
        'especially', 'during', 'tried', 'resetting', 'password', 'no change',
        'event viewer', 'application', 'view', 'folder', 'permissions',
        'might be incorrect', 'jobs', 'stuck', 'queue', 'model', 'hp', 'laserjet',
        'remote', 'video', 'conferences', 'getting', 'there', 'here', 'we', 'my',
        'me', 'them', 'they', 'their', 'we', 'our', 'out', 'up', 'down', 'about',
        'from', 'into', 'onto', 'upon', 'after', 'before', 'above', 'below', 'under',
        'over', 'through', 'within', 'without', 'among', 'across', 'behind', 'between',
        'beyond', 'along', 'around', 'near', 'off', 'on', 'out', 'up', 'with', 'without',
        'since', 'until', 'while', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
        'due to', 'related to', 'appears to be', 'seems to be', 'has been', 'is not', 'did not'
    ])

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove punctuation, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    # Keywords from 'Number' (exact match, high weight)
    if 'Number' in dataframe.columns:
        for num in dataframe['Number'].unique():
            if pd.notnull(num):
                all_keywords[str(num).upper()] += 10  # Strong weight for direct lookup

    # Keywords from 'Short description' (highest textual keyword impact)
    if 'Short description' in dataframe.columns:
        for desc in dataframe['Short description']:
            cleaned_desc = clean_text(desc)
            for word in cleaned_desc.split():
                if word and word not in stop_words and len(word) > 1:
                    all_keywords[word] += 5  # High weight

    # Keywords from 'Description' and 'Description_Customer'
    for col in ['Description', 'Description_Customer']:
        if col in dataframe.columns:
            for text in dataframe[col]:
                cleaned_text = clean_text(text)
                for word in cleaned_text.split():
                    if word and word not in stop_words and len(word) > 1:
                        all_keywords[word] += 2  # Standard weight

    # Keywords from 'Priority' (exact values)
    if 'Priority' in dataframe.columns:
        for prio in dataframe['Priority'].unique():
            if pd.notnull(prio):
                all_keywords[str(prio).lower()] += 3  # Medium weight

    # Keywords from 'Created' (general terms, not specific dates)
    # This field is more for filtering by date range, but we can extract general terms
    if 'Created' in dataframe.columns:
        for created_date in dataframe['Created']:
            if pd.notnull(created_date):
                # You might add logic here to extract "recent", "old", "today", "yesterday" if applicable
                # For now, specific dates are not treated as general search keywords for content.
                pass  # No keywords directly from dates for content search

    # Consider additional fields like 'Caller' for specific names/IDs if relevant for keyword search
    if 'Caller' in dataframe.columns:
        for caller in dataframe['Caller'].unique():
            if pd.notnull(caller) and str(caller).strip() != '':
                all_keywords[str(caller).lower()] += 2

    # Filter out very low frequency or irrelevant keywords
    # You can adjust the threshold based on your dataset size and desired granularity
    # Sorting for review, but for actual search, a set of unique keywords is often sufficient.
    # We convert to a set to remove duplicates and use for quick lookup later
    # Only take keywords that appear at least 'min_frequency' times, or have high initial weight
    min_frequency = 2  # Example threshold
    final_keywords_set = set([word for word, count in all_keywords.items() if
                              count >= min_frequency or count >= 5])  # Keep highly weighted ones regardless of frequency

    return list(final_keywords_set)  # Return as a list for consistency


# Generate keywords when the application starts
print("Generating comprehensive search keywords from the dataset...")
# Pass the global 'df' DataFrame to the function
generated_keywords = generate_comprehensive_keywords(df)
print(f"Generated {len(generated_keywords)} comprehensive keywords.")
# --- End Keyword Generation Logic ---


# Prepare RDF Graph
EX = Namespace("http://example.org/")
g = Graph()


def add_incident_to_graph(index, row):
    if 'Number' in row and pd.notnull(row['Number']):
        incident_uri = URIRef(f"http://example.org/incident/{row['Number']}")
        g.add((incident_uri, RDF.type, EX.Incident))
        for field in df.columns:
            if pd.notnull(row.get(field)):
                g.add((incident_uri, EX[field.replace(' ', '_')], Literal(str(row[field]))))
    else:
        print(f"Warning: Row {index} skipped due to missing 'Number' column.")


print("Building Knowledge Graph from dataset...")
for i, row in df.iterrows():
    add_incident_to_graph(i, row)
print(f"Knowledge Graph built with {len(g)} triples.")


# Build text for vector store
def row_to_text(row):
    # Ensure all specified fields are checked for existence and notnull
    fields_to_include = [
        'Number', 'Description', 'Description_Customer', 'Created',
        'Caller', 'Priority', 'Additional comments', 'Short description'
    ]
    main_parts = []
    for field in fields_to_include:
        if field in row and pd.notnull(row[field]):
            main_parts.append(str(row[field]))
    main = ' | '.join(main_parts)

    others_parts = []
    for col in df.columns:
        if col not in fields_to_include and col in row and pd.notnull(row[col]):
            others_parts.append(f"{col}: {row[col]}")
    others = ' | '.join(others_parts)

    return f"{main}\n\n{others}" if others else main


df['Document'] = df.apply(row_to_text, axis=1)

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = []
for doc in df['Document'].tolist():
    chunks.extend(splitter.split_text(doc))
print(f"Documents chunked into {len(chunks)} pieces.")

# Vector DB - Using GoogleGenerativeAIEmbeddings
print("Creating vector store with GoogleGenerativeAIEmbeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
print("Vector store created.")

# LLM - Using ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.5, max_output_tokens=10000)
print(f"LLM initialized with model: {llm.model}")


# Graph Query for context
def query_graph_for_context(prompt):
    terms = prompt.lower().split()
    matches = []
    for s, p, o in g:
        if any(term in str(o).lower() for term in terms) or \
                any(term in str(s).lower() for term in terms) or \
                any(term in str(p).lower() for term in terms):
            matches.append((s, p, o))

    summary = "\n".join([f"{str(s).split('/')[-1]} -> {str(p).split('/')[-1]}: {o}" for s, p, o in matches[:20]])
    return summary if summary else "No highly relevant knowledge graph facts found for this specific query."


# LLM Prompt Composition
def answer_query(prompt):
    keyword_match_context = ""
    # Use the dynamically generated keywords here
    keywords_to_search = generated_keywords  # Now using the keywords derived from your dataset

    found_keyword_in_prompt = False
    # Check if any part of the prompt matches any of the generated keywords
    # This check is more for deciding whether to trigger the direct keyword search on DataFrame
    # For a more nuanced approach, you'd parse the prompt for specific entities.
    for keyword in keywords_to_search:
        # Use regex to find whole words or specific phrases, case-insensitive
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', prompt.lower()):
            found_keyword_in_prompt = True
            break

    if found_keyword_in_prompt:
        # Instead of just using the raw prompt, split it and search for individual words/phrases
        # that are likely to be strong indicators from the prompt.
        # This is a basic approach; for complex prompts, N-gram matching or NLP entity extraction would be better.
        prompt_terms = [word for word in re.findall(r'\b\w+\b', prompt.lower()) if
                        word not in generated_stop_words_for_prompt_parsing]  # using a slightly different stop word list for prompt parsing

        # Build a regex pattern to find incidents that contain *any* of the significant prompt terms
        # and *any* of the generated keywords.
        # This is a simple OR search, which can be broad.
        # For a more precise search, you might require ALL significant prompt terms to be present.
        search_pattern_parts = [re.escape(term) for term in prompt_terms if term in keywords_to_search] + \
                               [re.escape(keyword) for keyword in keywords_to_search if
                                re.search(r'\b' + re.escape(keyword.lower()) + r'\b', prompt.lower())]

        # Ensure no empty patterns if no direct keyword match found in prompt terms
        if search_pattern_parts:
            search_regex = '|'.join(search_pattern_parts)
            # Search across the 'Document' column which combines all relevant fields
            keyword_results = df[df['Document'].str.contains(search_regex, case=False, na=False)]
        else:
            keyword_results = pd.DataFrame()  # No relevant terms to search for

        if not keyword_results.empty:
            total_matches = len(keyword_results)
            keyword_match_context += f"Found {total_matches} incidents across the dataset matching relevant terms from your query.\n"
            sample_incidents = keyword_results  # Keep all for now, can be limited if performance is an issue

            for i, row in sample_incidents.iterrows():
                incident_summary = f"Incident Number: {row.get('Number', 'N/A')}"
                # Dynamically add relevant fields to the summary
                for field in ['Short description', 'Created', 'Priority', 'Description_Customer', 'Description',
                              'Caller']:
                    if field in row and pd.notnull(row[field]):
                        value = str(row[field])
                        if field == 'Description' and len(value) > 100:
                            value = value[:100] + "..."  # Truncate long descriptions
                        incident_summary += f", {field.replace('_', ' ').title()}: {value}"
                keyword_match_context += incident_summary + "\n"
        else:
            keyword_match_context = "No specific incidents found via direct keyword search in the dataset for this query."

    else:
        keyword_match_context = "No direct keyword matches found for a specific incident search. Relying on semantic similarity and knowledge graph."

    vector_docs = vectorstore.similarity_search(prompt, k=100)
    vector_context = "\n\n".join([doc.page_content for doc in vector_docs])

    graph_context = query_graph_for_context(prompt)

    full_prompt = f"""You are a highly skilled ServiceNow Incident Analyst AI. Your primary goal is to provide concise, accurate, and helpful answers based ONLY on the provided information.

Here's the relevant information extracted from our ServiceNow Incident database:

Comprehensive Keyword Matches (if applicable, covering the entire dataset for specific terms):
{keyword_match_context}

Knowledge Graph Facts (derived from the entire dataset, showing relationships and specific attributes):
{graph_context}

Relevant Incident Details (semantically similar incidents retrieved from the dataset):
{vector_context}

Based *only* on the above information, please answer the following question. If the provided information does not contain the answer, clearly state that you do not have enough information to answer. Avoid making up information. When asked for a count, please provide the exact count if available in the "Comprehensive Keyword Matches" section, or state if it's not possible to determine from the given context.

Question:
{prompt}

Answer:"""

    try:
        response = llm.invoke(full_prompt)
        return response.content
    except Exception as e:
        return f"An error occurred while calling the AI model: {e}. Please check your API key, model availability, or try a different query."


# Global stop words for prompt parsing (distinct from keyword generation stop words if needed)
generated_stop_words_for_prompt_parsing = set([
    'what', 'is', 'are', 'the', 'tell', 'me', 'about', 'most', 'frequent', 'by', 'caller', 'or', 'incidents',
    'with', 'can', 'you', 'give', 'how', 'many', 'show', 'list', 'of', 'please', 'any', 'information', 'regarding',
    'for', 'get', 'details'
])


# Web UI with Flask
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    user_prompt = ""
    if request.method == "POST":
        user_prompt = request.form["prompt"]
        answer = answer_query(user_prompt)

    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Incident Gemini AI Analyst</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #eef2f7; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .container { max-width: 850px; margin-top: 60px; background: #fff; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
        .form-control, .btn-primary { font-size: 1rem; padding: 10px 15px; }
        .result-box { margin-top: 25px; background: #f9fafc; border-left: 5px solid #0d6efd; padding: 20px; border-radius: 10px; font-family: 'Consolas', 'Courier New', monospace; white-space: pre-wrap; overflow-x: auto; line-height: 1.6; }
        .title { text-align: center; font-weight: bold; margin-bottom: 30px; color: #333; }
        textarea { resize: vertical; min-height: 100px; }
        .btn-primary { background-color: #0d6efd; border-color: #0d6efd; transition: background-color 0.2s ease; }
        .btn-primary:hover { background-color: #0b5ed7; border-color: #0a58ca; }
    </style>
</head>
<body>
<div class="container">
    <h2 class="title">🔍 Ask Your ServiceNow Incident Data with Gemini AI</h2>
    <form method="POST">
        <div class="mb-3">
            <label for="prompt" class="form-label">Your Question:</label>
            <textarea class="form-control" name="prompt" rows="4" placeholder="e.g., What are the most frequent issues by Caller? Or tell me more about incidents with DUPLICATECHECK" required>{{ request.form.get('prompt', '') }}</textarea>
        </div>
        <button type="submit" class="btn btn-primary w-100">Get Answer from OmniOps@AI (Gemini)</button>
    </form>
    {% if answer %}
    <div class="result-box"><h5>📋 Answer</h5><p>{{ answer }}</p></div>
    {% endif %}
</div>
</body>
</html>
""", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)