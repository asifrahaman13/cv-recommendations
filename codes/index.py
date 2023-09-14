import os
import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel

import json
data={}
with open("cv_data.json", "r") as file:
    data=json.load(file)

# print(data)


company_and_job_desctiptions={}

for idx, item in enumerate(data):
    # print("*************************************************************************************************************8")
    # for i in range(5):
    #     print()
    # print(item['company_name'])
    # print(item['job_description'])
    company_and_job_desctiptions[item['company_name']]=item['job_description']
# print(company_and_job_desctiptions)



# for key, value in company_and_job_desctiptions.items():
#     print(value)

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Load job descriptions and extracted CV details from JSON
with open('extracted_details.json', 'r') as json_file:
    cv_details = json.load(json_file)

# Sample job descriptions (replace with your own data)
# job_descriptions = [
#     "We are looking for a software engineer with expertise in Python and machine learning.",
#     "Seeking a project manager with a background in construction management.",
#     "Hiring a data scientist with experience in deep learning and natural language processing.",
# ]

job_descriptions =[]
for key, value in company_and_job_desctiptions.items():
    job_descriptions.append(value)


# Initialize a dictionary to store top 5 CVs for each job description
top_5_cvs = {}

# Tokenize and embed job descriptions
job_desc_embeddings = [model(**tokenizer(job_desc, return_tensors='pt', padding=True, truncation=True, max_length=512)).last_hidden_state.mean(dim=1) for job_desc in job_descriptions]

# Tokenize and embed CV details
collected_cvs = {job_desc: [] for job_desc in job_descriptions}

# Tokenize and embed CV details
for cv in cv_details:
    # Iterate over job descriptions
    for job_desc in job_descriptions:
        cv_text = f"{cv['Category']} {', '.join(cv['Skills'])} {', '.join(cv['Education'])}"
        cv_embedding = model(**tokenizer(cv_text, return_tensors='pt', padding=True, truncation=True, max_length=512)).last_hidden_state.mean(dim=1)

        # Calculate cosine similarity between job descriptions and CVs using PyTorch
        cv_embedding = cv_embedding.squeeze(0)  # Remove the batch dimension
        job_desc_embedding = job_desc_embeddings[job_descriptions.index(job_desc)].squeeze(0)  # Get the corresponding job description embedding

        similarity = torch.nn.functional.cosine_similarity(job_desc_embedding, cv_embedding, dim=0).item()

        # Store the CV and similarity score
        collected_cvs[job_desc].append((cv['PDFFilename'], similarity))

# Sort the collected CVs by similarity score and select the top 5
for job_desc, cvs in collected_cvs.items():
    top_5_cvs[job_desc] = sorted(cvs, key=lambda x: x[1], reverse=True)[:5]

def find_the_key(job_desctiption):
    for key, value in  company_and_job_desctiptions.items():
        if(value==job_desctiption):
            return key

# Print the top 5 CVs for each job description
for job_desc, cvs in top_5_cvs.items():
    company=find_the_key(job_desc)
    print(f"Top 5 CVs for '{company}':")
    for cv, similarity in cvs:
        print(f"CV: {cv}, Similarity Score: {similarity}")
    print()
