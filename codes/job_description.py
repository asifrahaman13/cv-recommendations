import os 
PATH="archive/data/data"
items = os.listdir(PATH)
directories = [item for item in items if os.path.isdir(os.path.join(PATH, item))]

print("Directories in the path:")
list_of_categories=[]
for directory in directories:
    list_of_categories.append(directory)
    # print(directory)
print(list_of_categories)

import os
import pdfplumber
import json

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
