from flask import Flask, render_template_string, request
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os
import tiktoken

app = Flask(__name__)

# Set OpenAI API Key
OPENAI_API_KEY = '***'  # 🔐 Replace with your actual OpenAI API key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Load and preprocess dataset
dataset_path = 'C:\\Users\\s0s0cqc\\Downloads\\incident.xlsx'
df = pd.read_excel(dataset_path, engine='openpyxl')
df = df.dropna(axis=1, how='all')

def row_to_text(row):
    fields = ['Number', 'Description', 'Description_Customer', 'Caller', 'Impact',
              'Assignment group', 'Priority', 'Additional comments', 'Created',
              'Created by', 'Issue start time']
    text = ""
    for field in fields:
        if pd.notnull(row.get(field)):
            text += f"{field}: {row[field]}\n"
    return text.strip()

df['Document'] = df.apply(row_to_text, axis=1)
print(df['Document'])
# Token counter
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def count_tokens(text):
    return len(encoding.encode(text))

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_chunks = []
for doc_text in df['Document'].tolist():
    chunks = splitter.split_text(doc_text)
    for chunk in chunks:
        if count_tokens(chunk) <= 3000:
            all_chunks.append(chunk)
        else:
            truncated = encoding.decode(encoding.encode(chunk)[:3000])
            all_chunks.append(truncated)

# Embedding and Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(all_chunks, embedding=embeddings)

# LLM Initialization
llm = ChatOpenAI(temperature=0.5, max_tokens=2048)

# Prompt handler
def answer_query(prompt):
    docs = vectorstore.similarity_search(prompt, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    full_prompt = f"""You are an expert IT analyst. Use the below incident data to answer the question.

Incident data:
{context}

Question:
{prompt}

Answer:"""
    response = llm.invoke(full_prompt)
    return response.content

# Web UI route
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        user_prompt = request.form["prompt"]
        answer = answer_query(user_prompt)

    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ServiceNow Incident GPT</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #eef2f7;
        }
        .container {
            max-width: 850px;
            margin-top: 60px;
            background: #fff;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .form-control {
            font-size: 1rem;
        }
        .result-box {
            margin-top: 25px;
            background: #f9fafc;
            border-left: 5px solid #0d6efd;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        .result-box h5 {
            margin-bottom: 15px;
            color: #0d6efd;
            font-weight: bold;
        }
        .btn-primary {
            font-size: 1rem;
        }
        .title {
            text-align: center;
            font-weight: bold;
            margin-bottom: 30px;
        }
        textarea {
            resize: vertical;
        }
    </style>
</head>
<body>
<div class="container">
    <h2 class="title">🔍 Ask Your ServiceNow Data Anything</h2>
    <form method="POST">
        <div class="mb-3">
            <label for="prompt" class="form-label">Your Question:</label>
            <textarea class="form-control" name="prompt" rows="4" placeholder="e.g., What are the most frequent incident types for the Fintech department?"></textarea>
        </div>
        <button type="submit" class="btn btn-primary w-100">Ask OmniOps@Ai</button>
    </form>

    {% if answer %}
    <div class="result-box">
        <h5>📋 Answer</h5>
        <p>{{ answer }}</p>
    </div>
    {% endif %}
</div>
</body>
</html>
    """, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
