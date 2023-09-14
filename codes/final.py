# %%

from datasets import load_dataset
import json
import os 
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import pdfplumber

# %%
dataset = load_dataset("jacob-hugging-face/job-descriptions")

# %%
print(dataset['train'].to_dict())

# %% [markdown]
# Convert the dictionry data into json format

# %%
def transform_to_horizontal(data, fields, limit=None):
    horizontal_data = []

    for idx, item in enumerate(data):
        if limit is not None and idx >= limit:
            break

        horizontal_item = {field: item[field] for field in fields}
        horizontal_data.append(horizontal_item)

    return horizontal_data

# Specify the fields you want to include in the horizontal format
fields_to_include = ["company_name", "job_description", "position_title", "description_length", "model_response"]

# Transform the data to horizontal format
horizontal_data = transform_to_horizontal(dataset["train"], fields_to_include, limit=15)

# Saving the horizontal data to a JSON file
with open("job_descriptions/cv_data.json", "w") as file:
    json.dump(horizontal_data, file)


# %% [markdown]
# Perform data extraction from the pdf.

# %%

PATH="archive/data/data"
items = os.listdir(PATH)
directories = [item for item in items if os.path.isdir(os.path.join(PATH, item))]

print("Directories in the path:")
list_of_categories=[]
for directory in directories:
    list_of_categories.append(directory)
    # print(directory)
print(list_of_categories)


# Function to extract category, skills, and education from a PDF
def extract_details(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Initialize variables to store extracted details
            category = None
            skills = []
            education = []

            # Iterate through pages in the PDF
            for page in pdf.pages:
                text = page.extract_text()

                # Search for patterns in the extracted text
                if "Category" in text:
                    category = text.split("Category")[1].strip()
                if "Skills" in text:
                    skills = [skill.strip() for skill in text.split("Skills")[1].split(",")]
                if "Education" in text:
                    education = [edu.strip() for edu in text.split("Education")[1].split(";")]

            return {
                'PDFFilename': os.path.basename(pdf_path), # Include the PDF filename
                'Category': category,
                'Skills': skills,
                'Education': education,
            }
    except Exception as e:
        print(f"Error extracting details from {pdf_path}: {str(e)}")
        return None

# Create a list to store extracted details
all_details = []

for directory in list_of_categories:
    # Directory containing PDF CVs
    pdf_directory = f'archive/data/data/{directory}'
    # Iterate through PDF files and extract details
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            details = extract_details(pdf_path)
            if details:
                print(f"Details extracted from {filename}:\n{details}\n")
                all_details.append(details)

# Save the extracted details in a JSON file
output_file = 'extracted/extracted_details.json'
with open(output_file, 'w') as json_file:
    json.dump(all_details, json_file, indent=4)

print(f"Extracted details saved to {output_file}")


# %% [markdown]
# Tokenize and preprocessing

# %%
# Free GPU memory
torch.cuda.empty_cache()

# %%

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Free GPU memory
torch.cuda.empty_cache()

# Load your CV data from cv_data.json
data = {}
with open("job_descriptions/cv_data.json", "r") as file:
    data = json.load(file)
    file.close()

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
    json_file.close()

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


# %%
# Print the top 5 CVs for each job description

shortlisted_cvs={}
for job_desc, cvs in top_5_cvs.items():
    company = find_the_key(job_desc)
    print(f"Top 5 CVs for '{company}':")
    list_of_selected_resumes=[]
    for cv, similarity in cvs:
        print(f"CV: {cv}, Similarity Score: {similarity}")
        list_of_selected_resumes.append(cv)
    shortlisted_cvs[company]=list_of_selected_resumes
    print()

# %%
print(shortlisted_cvs)

# %% [markdown]
# Save the cvs of the final shortlisted candidates

# %%
with open("shortlisted/shortlisted_cvs.json", "w") as file:
    json.dump(shortlisted_cvs, file, indent=4)
    file.close()



# %%
