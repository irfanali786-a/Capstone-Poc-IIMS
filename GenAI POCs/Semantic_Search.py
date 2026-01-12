from flask import Flask, render_template_string
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import pandas as pd

app = Flask(__name__)

# API Key
OPENAI_API_KEY = '***'  # Replace with your actual key or use environment variable

# Load dataset
dataset_path = 'C:\\Users\\s0s0cqc\\Downloads\\email_thread_summaries.csv'
df = pd.read_csv(dataset_path)

# Convert each summary into a document and split individually
documents = df['summary'].tolist()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # Adjusted chunk size and overlap

# Split each summary separately and flatten the list
text_chunks = []
for doc in documents:
    chunks = text_splitter.split_text(str(doc))
    text_chunks.extend(chunks)

# Create embeddings and FAISS index
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Load the LLM with adjusted max tokens
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, max_tokens=3000)  # Adjusted max_tokens to avoid exceeding limits

# Search function — just call vectorstore once per query
def search(query):
    return vectorstore.similarity_search(query, k=5)

@app.route('/')
def home():
    query = "what is the financial report for this year"
    results = search(query)

    # Render results in a visually appealing HTML table
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f4f4f9;
            }
            h1 {
                text-align: center;
                color: #333;
            }
            table {
                width: 80%;
                margin: 20px auto;
                border-collapse: collapse;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                background-color: #fff;
            }
            th, td {
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:hover {
                background-color: #f1f1f1;
            }
            .container {
                text-align: center;
            }
            .button {
                display: inline-block;
                margin: 10px;
                padding: 10px 20px;
                font-size: 16px;
                color: white;
                background-color: #4CAF50;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Search Results</h1>
        <div class="container">
            <a href="/" class="button">New Search</a>
        </div>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Content</th>
                </tr>
            </thead>
            <tbody>
                {% for i, doc in enumerate(results, 1) %}
                <tr>
                    <td>{{ i }}</td>
                    <td>{{ doc.page_content }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html_template, results=results, enumerate=enumerate)

if __name__ == '__main__':
    app.run(debug=True)
