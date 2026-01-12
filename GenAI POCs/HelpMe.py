#!/usr/bin/env python
# coding: utf-8
# this is a code for email scanner LLM

# In[1]:


import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Removed Kaggle API initialization and dataset download.
# Updated the dataset path to read directly from the specified file.
dataset_path = 'C:\\Users\\s0s0cqc\\Downloads\\email_thread_summaries.csv'
emails_df = pd.read_csv(dataset_path)
emails_text = emails_df['summary'].tolist()

# Chunking strategy - using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, max_df=0.85, min_df=2)
email_vectors = vectorizer.fit_transform(emails_text)

# Check if the data is loaded and vectorized
# Print sample email summaries and the shape of the TF-IDF matrix
print("Sample email summaries:", emails_df['summary'].head())
print("TF-IDF matrix shape:", email_vectors.shape)

# Import PyTorch and print its version
import torch
print(torch.__version__)

# Import cosine similarity for search layer
from sklearn.metrics.pairwise import cosine_similarity

# Define sample queries for searching email summaries
queries = [
    "Find information about project updates in emails from last month.",
    "Retrieve emails related to financial reports from Q3 last year.",
    "Search for discussions on customer feedback in the emails."

]

# Embedding the queries using the same vectorizer
query_vectors = vectorizer.transform(queries)

# Search and retrieve top results
top_results = []
for query_vec in query_vectors:
    similarities = cosine_similarity(query_vec, email_vectors)
    top_idx = similarities.argsort()[0][-3:][::-1]  # Get indices of top 3 results
    top_results.append(emails_df.iloc[top_idx]['summary'].tolist())

# Print or process top results
for i, query in enumerate(queries):
    print(f"Query: {query}")
    for idx, result in enumerate(top_results[i]):
        print(f"Result {idx + 1}: {result[:200]}...")
    print()

# In[4]:


# Generation Layer using Hugging Face Transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Sample prompts
prompts = [
    "Generate a summary of the project updates discussed in the emails from last month.",
    "Explain the findings from the financial reports in Q3 last year based on the emails.",
    "Provide insights from customer feedback discussions found in the emails."

]

# Generate responses
generated_outputs = []
for prompt in prompts:
    print(f"Prompt: {prompt}")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"Input IDs: {input_ids}")

    # Set pad_token_id to eos_token_id to avoid warning
    output = model.generate(
        input_ids,
        max_length=300,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated Output: {generated_output}")
    generated_outputs.append(generated_output)
    print()

# Print or process generated outputs
for i, prompt in enumerate(prompts):
    print(f"Prompt: {prompt}")
    print(f"Generated Output: {generated_outputs[i]}")
    print()

#
