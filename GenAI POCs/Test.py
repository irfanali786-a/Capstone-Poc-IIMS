#############################################
# OmniOps@AI Incident RCA Assistant (Flask)
#############################################

# --- macOS / libomp safety guards (must be first!) ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import re
import httpx
import pandas as pd
from flask import Flask, render_template_string, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
import tiktoken


#############################################
# 1. CONFIG
#############################################
AZURE_OPENAI_API_KEY="eyJzZ252ZXIiOiIxIiwiYWxnIjoiSFMyNTYiLCJ0eXAiOiJKV1QifQ.eyJqdGkiOiI2MzM2Iiwic3ViIjoiMTQ5NyIsImlzcyI6IldNVExMTUdBVEVXQVktU1RHIiwiYWN0IjoibjBiMDZ2byIsInR5cGUiOiJBUFAiLCJpYXQiOjE3NjA2OTE4MjEsImV4cCI6MTc3NjI0MzgyMX0.qo34o05_7J3cvTe1cgk48QUaCoaBkxWcRRZxuoRNm_4"
AZURE_OPENAI_ENDPOINT="https://wmtllmgateway.stage.walmart.com/wmtllmgateway/"   # ends with /
AZURE_OPENAI_API_VERSION="2024-10-21"                  # or your version
AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o@2024-11-20"           # your DEPLOYMENT name
AZURE_OPENAI_EMBED_DEPLOYMENT="text-embedding-3-large"

DATASET_PATH = '/Users/n0b06vo/Downloads/incident.xlsx' # replace with your Path


#############################################
# 2. HTTP CLIENT SHARED BY AZURE CLIENTS
#############################################

http_client = httpx.Client(verify=False, timeout=30)


#############################################
# 3. FLASK APP
#############################################

app = Flask(__name__)


#############################################
# 4. TOKENIZER HELPERS
#############################################

# Get encoding safely. Some environments may not know "gpt-5-mini".
try:
    encoding = tiktoken.encoding_for_model("gpt-5-mini")
except Exception:
    encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text or ""))

def truncate_tokens(text: str, max_tokens: int = 3000) -> str:
    tokens = encoding.encode(text or "")
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)


#############################################
# 5. FIELDS OF INTEREST FOR METADATA
#############################################

# These are the structured fields we want to carry as metadata and show to the LLM.
FIELDS_OF_INTEREST = [
    "Number",
    "Description",
    "Description_Customer",
    "Short description (Knowledge search)",
    "Caller",
    "Impact",
    "Assignment group",
    "Priority",
    "Additional comments",  # raw, but we'll also add a cleaned summary version
    "Created",
    "Created by",
    "Issue start time",
]


#############################################
# 6. COMMENT CLEANING / SUMMARIZATION
#############################################

def clean_additional_comments(raw: str, max_points: int = 4) -> str:
    """
    Takes the raw 'Additional comments' timeline field (with timestamps,
    user IDs, attachments, etc.) and returns a short bullet-style summary
    for metadata / LLM readability.

    Example input block:
    "03-17-2025 23:43:41 - Omkar ... This was an intermittent failure ...\n\n
     03-17-2025 14:19:32 - Saravanakumar ... Monitoring the subsequent runs ..."

    Output:
    "- Intermittent failure; downstream succeeded in next run.
     - Monitoring subsequent runs; appears transient.
     - System warning alert 53606868 on Central Item Level Ledger from CILL-Finance."
    """

    if not isinstance(raw, str):
        return ""

    # Split by blank-line gaps between updates
    # This assumes your timeline uses blank lines between entries,
    # which matches your sample.
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    cleaned_points = []

    for b in blocks:
        # Remove leading "MM-DD-YYYY HH:MM:SS - <person> (...) (Additional comments)"
        # We'll try to be flexible (.*?\(Additional comments\))
        b_nolead = re.sub(
            r"^\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}\s+-\s+.*?\(Additional comments\)\s*",
            "",
            b,
            flags=re.IGNORECASE,
        )

        # Collapse multiple spaces/newlines
        b_nolead = re.sub(r"\s+", " ", b_nolead).strip()

        # Drop pure attachment noise if you want
        # e.g. "Attachment: Screenshot blah blah"
        if b_nolead.lower().startswith("attachment:"):
            continue

        if b_nolead:
            cleaned_points.append(b_nolead)

    if not cleaned_points:
        return ""

    # Keep only first few points so prompt doesn't explode
    cleaned_points = cleaned_points[:max_points]

    # Bulletize them
    bulletized = "- " + "\n- ".join(cleaned_points)
    return bulletized


#############################################
# 7. LOAD DATA + BUILD VECTORSTORE (RUNS AT STARTUP)
#############################################

# Read the Excel
df = pd.read_excel(DATASET_PATH, engine="openpyxl")
df = df.dropna(axis=1, how="all")

# Splitter for long comment timelines
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)

texts = []       # what FAISS will embed/search
metadatas = []   # structured fields we’ll attach and later show in RCA prompt

for _, row in df.iterrows():
    # Raw short desc + comments
    short_text = str(row.get("Short description (Knowledge search)", "")).strip()
    raw_comments = str(row.get("Additional comments", "")).strip()

    # Build combined text that FAISS will search over
    combined_text = (
        f"Short Description: {short_text}\n\n"
        f"Comments Timeline:\n{raw_comments}"
    ).strip()

    if not combined_text:
        # skip empty incidents
        continue

    # Build structured metadata for this incident.
    # base_meta includes your main columns,
    # plus a summary-cleaned "Additional comments" field for readability.
    base_meta = {}
    for f in FIELDS_OF_INTEREST:
        val = row.get(f)
        if pd.notnull(val):
            base_meta[f] = str(val)
        else:
            base_meta[f] = ""

    base_meta["Additional comments (summary)"] = clean_additional_comments(raw_comments)

    # Chunk the combined text (this handles long timelines)
    chunks = splitter.split_text(combined_text)

    for chunk in chunks:
        safe_chunk = truncate_tokens(chunk, max_tokens=3000)
        texts.append(safe_chunk)
        metadatas.append(base_meta)

# Create the embeddings client
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",                 # model family
    azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT, # your Azure embedding deployment name
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    http_client=http_client,
)

# Create the FAISS vector index
# vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

import time
from tqdm import tqdm   # optional, for progress bar (pip install tqdm)
from pathlib import Path

SAVE_PATH = "faiss_index"

if Path(SAVE_PATH).exists():
    print("📂 Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("🚀 Building FAISS index (first run)...")
    # <insert the batching loop here>

    BATCH_SIZE = 10
    ALL_CHUNKS = []
    ALL_METAS = []

    print(f"Total chunks to embed: {len(texts)}")

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_metas = metadatas[i : i + BATCH_SIZE]

        # embed this batch
        vs_batch = FAISS.from_texts(batch_texts, embedding=embeddings, metadatas=batch_metas)
        ALL_CHUNKS.append(vs_batch)

        # 💤 small pause so you stay under rate limit
        time.sleep(2)

    # merge batches into one FAISS index
    vectorstore = ALL_CHUNKS[0]
    for vs in ALL_CHUNKS[1:]:
        vectorstore.merge_from(vs)

    print("✅ Embeddings complete, FAISS index built.")
    SAVE_PATH = "faiss_index"
    vectorstore.save_local(SAVE_PATH)
    print(f"💾 Saved FAISS index to {SAVE_PATH}")


# Create the chat LLM client
llm = AzureChatOpenAI(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT,     # your Azure chat deployment name
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.5,
    max_tokens=2048,
    http_client=http_client,
)


#############################################
# 8. CORE QUERY / RCA BUILDER
#############################################

def answer_query(user_question: str):
    """
    1. Retrieve most similar past incidents from FAISS.
    2. Build an RCA prompt using structured metadata + snippet text.
    3. Ask the LLM for Root Cause / Evidence / Confidence.
    4. Return (debug_prompt_sent_to_model, model_answer_text)
    """
    try:
        # Step A: retrieve k most similar incidents
        docs = vectorstore.similarity_search(user_question, k=5)
    except Exception as e:
        return (
            f"[ERROR retrieving docs]\n\nUser Question:\n{user_question}",
            f"RCA unavailable because retrieval failed: {e}",
        )

    # Step B: format each retrieved doc into a block, to feed the LLM
    incident_blocks = []

    for i, doc in enumerate(docs, start=1):
        md = doc.metadata or {}

        # Build a human/LLM friendly section
        lines = []
        lines.append(f"# Incident {i}")
        lines.append(f"Number: {md.get('Number', '')}")
        lines.append(f"Assignment group: {md.get('Assignment group', '')}")
        lines.append(f"Priority: {md.get('Priority', '')}")
        lines.append(f"Issue start time: {md.get('Issue start time', '')}")
        lines.append(f"Short description: {md.get('Short description (Knowledge search)', '')}")
        lines.append("Additional comments (summary):")
        lines.append(md.get("Additional comments (summary)", ""))

        # Also include actual retrieved text chunk (raw timeline snippet).
        # This gives the LLM evidence and deep detail.
        lines.append("\nSnippet from timeline chunk:")
        lines.append(doc.page_content)

        # Join lines for this incident block
        incident_blocks.append("\n".join(l for l in lines if l is not None))

    # Join all 5 incidents into one big context
    context_block = "\n\n---\n\n".join(incident_blocks)

    # Step C: Construct final LLM prompt
    full_prompt = f"""You are an expert IT incident analyst.

Using ONLY the 'Previous Similar Incidents' below, analyze the 'Current Incident' and produce a Root Cause Analysis (RCA).

Rules:
- Base your answer strictly on provided incidents and fields.
- If the cause cannot be determined from the available information, respond with "I don't know".
- Do not invent, guess, or rely on outside knowledge.
- Do not claim anything that is not supported by the evidence.

Respond in this exact structure:
- Root Cause:
- Supporting Evidence:
- Confidence Level (High / Medium / Low):

Previous Similar Incidents:
{context_block}

Current Incident:
{user_question}

Answer:"""

    # Step D: Call LLM
    try:
        llm_response = llm.invoke(full_prompt)
        answer_text = getattr(llm_response, "content", str(llm_response))
    except Exception as e:
        answer_text = f"LLM call failed: {e}"

    return (full_prompt, answer_text)


#############################################
# 9. FLASK ROUTE / UI
#############################################

@app.route("/", methods=["GET", "POST"])
def index():
    model_prompt_for_debug = ""
    model_answer = ""

    if request.method == "POST":
        user_prompt = request.form.get("prompt", "").strip()
        model_prompt_for_debug, model_answer = answer_query(user_prompt)

    # Note:
    # We render results in <pre> blocks so newlines are preserved.
    # Jinja escapes HTML by default in {{ }} so we're safe.
    return render_template_string(
        """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OmniOps@AI - Incident RCA Assistant</title>
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
        .title {
            text-align: center;
            font-weight: 600;
            margin-bottom: 30px;
        }
        textarea.form-control {
            resize: vertical;
            min-height: 120px;
            font-size: 0.95rem;
        }
        .btn-primary {
            font-size: 1rem;
            font-weight: 500;
        }
        .result-box {
            margin-top: 25px;
            background: #f9fafc;
            border-left: 5px solid #0d6efd;
            padding: 20px;
            border-radius: 10px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            overflow-x: auto;
        }
        .result-box h5 {
            margin-bottom: 15px;
            color: #0d6efd;
            font-weight: bold;
        }
        pre {
            white-space: pre-wrap;
            word-break: break-word;
            margin: 0;
        }
    </style>
</head>
<body>
<div class="container">
    <h2 class="title">🔍 OmniOps@AI — RCA from ServiceNow History</h2>
    <form method="POST">
        <div class="mb-3">
            <label for="prompt" class="form-label">Describe the current incident / ask a question:</label>
            <textarea class="form-control" id="prompt" name="prompt"
                placeholder="Example: Payment API timing out for fintech users in us-west after deploy 14.2. Is there any known RCA?"></textarea>
        </div>
        <button type="submit" class="btn btn-primary w-100">Analyze</button>
    </form>

    {% if model_answer %}
    <div class="result-box">
        <h5>🔎 Model Input Context</h5>
        <pre>{{ model_prompt_for_debug }}</pre>
    </div>

    <div class="result-box">
        <h5>📋 RCA Answer</h5>
        <pre>{{ model_answer }}</pre>
    </div>
    {% endif %}
</div>
</body>
</html>
        """,
        model_prompt_for_debug=model_prompt_for_debug,
        model_answer=model_answer,
    )


#############################################
# 10. MAIN
#############################################

if __name__ == "__main__":
    # Note: debug=True restarts the server with a reloader,
    # which means all the FAISS-building code above will run twice.
    # For dev it's fine, just be aware. For prod, run with gunicorn and debug=False.
    app.run(debug=True)
