import os
import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your CV data from cv_data.json
data = {}
with open("job_descriptions/cv_data.json", "r") as file:
    data = json.load(file)

# Extract company names and job descriptions
company_and_job_descriptions = {}
for item in data:
    company_and_job_descriptions[item['company_name']] = item['job_description']

# Load DistilBERT tokenizer and model on the GPU
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# Load job descriptions and extracted CV details from JSON
with open('extracted/extracted_details.json', 'r') as json_file:
    cv_details = json.load(json_file)

# Create a list of job descriptions
job_descriptions = list(company_and_job_descriptions.values())

# Initialize a dictionary to store top 5 CVs for each job description
top_5_cvs = {}

# Tokenize and embed job descriptions
job_desc_embeddings = [model(**tokenizer(job_desc, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)).last_hidden_state.mean(dim=1) for job_desc in job_descriptions]

# Initialize a dictionary to store collected CVs for each job description
collected_cvs = {job_desc: [] for job_desc in job_descriptions}

# Tokenize and embed CV details
for cv in cv_details:
    # Iterate over job descriptions
    for job_desc in job_descriptions:
        cv_text = f"{cv['Category']} {', '.join(cv['Skills'])} {', '.join(cv['Education'])}"
        cv_embedding = model(**tokenizer(cv_text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)).last_hidden_state.mean(dim=1)

        # Calculate cosine similarity between job descriptions and CVs using PyTorch
        cv_embedding = cv_embedding.squeeze(0)  # Remove the batch dimension
        job_desc_embedding = job_desc_embeddings[job_descriptions.index(job_desc)].squeeze(0)  # Get the corresponding job description embedding

        similarity = torch.nn.functional.cosine_similarity(job_desc_embedding, cv_embedding, dim=0).item()

        # Store the CV and similarity score
        collected_cvs[job_desc].append((cv['PDFFilename'], similarity))

# Sort the collected CVs by similarity score and select the top 5
for job_desc, cvs in collected_cvs.items():
    top_5_cvs[job_desc] = sorted(cvs, key=lambda x: x[1], reverse=True)[:5]

# Function to find the key (company name) for a given job description
def find_the_key(job_description):
    for key, value in company_and_job_descriptions.items():
        if value == job_description:
            return key

# Print the top 5 CVs for each job description
for job_desc, cvs in top_5_cvs.items():
    company = find_the_key(job_desc)
    print(f"Top 5 CVs for '{company}':")
    for cv, similarity in cvs:
        print(f"CV: {cv}, Similarity Score: {similarity}")
    print()
