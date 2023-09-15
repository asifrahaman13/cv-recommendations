# Import all the necessary libraries and packages in the code.
import os
import json
import torch
import pdfplumber
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeviceInteractions:
    # Function to empty the graphics memory
    def empty_graphics_card():
        if torch.cuda.is_available():
            # Free GPU memory
            torch.cuda.empty_cache()
        else:
            pass


# Create class for the Company Fields.
class CompanyFields:
    def __init__(self):
        self.dataset = load_dataset("jacob-hugging-face/job-descriptions")
        company_fields = list(self.dataset["train"].to_dict().keys())
        self.company_fields = company_fields

    def get_company_fields(self):
        return self.company_fields

    def transform_to_horizontal(self, data, fields, limit=None):
        # Initialzie list to hold the dataq.
        horizontal_data = []

        # Use teh enumerate method to get the incdex as well as the item.
        for idx, item in enumerate(data):
            # Break the loop if it crosses the limit.
            if limit is not None and idx >= limit:
                break
                # Append the items into the dictionary.
            horizontal_item = {field: item[field] for field in fields}
            # Append the dictionary data as elements of the list.
            horizontal_data.append(horizontal_item)

        # Return the list object.
        return horizontal_data

    # Fumction to convert the data into json format.
    def convert_to_json_format(self):
        # Transform the data to horizontal format
        self.horizontal_data = self.transform_to_horizontal(
            self.dataset["train"], self.get_company_fields(), limit=15
        )
        return self.horizontal_data

    # Save the job description data in json format.
    def save_job_description_data(self):
        # Saving the horizontal data to a JSON file
        with open("job_descriptions/cv_data.json", "w") as file:
            # Dump the json data into a file.
            json.dump(self.horizontal_data, file)
            # Its always a good practice to close the file after the operations are done.
            file.close()


class CvDataExtraction:
    """Extract Details From The CVs. The following codes helps to extract meaningful information from the PDFs. This may not be 100% accurate but contains reasonable and meaningful information from the CVS. After the extraction process is complete the data is stored in json format in the extracted_details.json file of the extracted folder."""

    def __init__(self, PATH):
        self.PATH = PATH

    def extract_details(self, pdf_path, category):
        # Function to extract category, skills, and education from a PDF
        print(pdf_path)
        # Wrap the code in try catch block to avoid any error.
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Initialize variables to store extracted details
                experience = None
                skills = []
                education = []

                # Iterate through pages in the PDF
                for page in pdf.pages:
                    text = page.extract_text()

                    # Search for patterns in the extracted text
                    if "Experience" in text:
                        experience = text.split("Experience")[1].strip()
                    if "Skills" in text:
                        skills = [
                            skill.strip()
                            for skill in text.split("Skills")[1].split(",")
                        ]
                    if "Education" in text:
                        education = [
                            edu.strip() for edu in text.split("Education")[1].split(";")
                        ]
                    # Return the data in the form of dictionary.
                return {
                    "PDFFilename": os.path.basename(
                        pdf_path
                    ),  # Include the PDF filename
                    "Category": category,
                    "Experience": experience,
                    "Skills": skills,
                    "Education": education,
                }
            # Create an exception and return None in that case.
        except Exception as e:
            print(f"Error extracting details from {pdf_path}: {str(e)}")
            return None

    def extract_details_from_cv(self):
        PATH = self.PATH
        # Extract all the directories under the mentioned path using list comprehension method.
        items = os.listdir(PATH)
        directories = [
            item for item in items if os.path.isdir(os.path.join(PATH, item))
        ]

        print("Directories in the path:")

        # Initialize a list which will hold the list of all the categories (basically the names of the directories)
        list_of_categories = []

        for directory in directories:
            list_of_categories.append(directory)
            # print(directory)
        print(list_of_categories)

        # Create a list to store extracted details
        all_details = []

        for directory in list_of_categories:
            # Directory containing PDF CVs
            pdf_directory = f"archive/data/data/{directory}"
            # Iterate through PDF files and extract details
            for filename in os.listdir(pdf_directory):
                if filename.endswith(".pdf"):
                    pdf_path = os.path.join(pdf_directory, filename)
                    details = self.extract_details(pdf_path, directory)
                    if details:
                        print(f"Details extracted from {filename}:\n{details}\n")
                        all_details.append(details)

        # Save the extracted details in a JSON file
        output_file = "extracted/extracted_details.json"
        with open(output_file, "w") as json_file:
            # Write the json data into a file.
            json.dump(all_details, json_file, indent=4)

            # Close the file.
            json_file.close()
        # Print the output file
        print(f"Extracted details saved to {output_file}")

    def load_job_description_data(self, json_file_path):
        # Load your CV data from cv_data.json
        data = {}
        with open(f"{json_file_path}", "r") as file:
            # Load json data from the file.
            data = json.load(file)
            # Close the file.
            file.close()
        return data


class CvSeclection:
    """
    Tokenize and preprocessing along with the similarity matrix.Find The Similarity Matrix Between The CVs And Company Description. In the following code we have used the PyTorch to find the cosine similarity matrix between the CVs and company description. Only the relevant CVs are selected for the process. The CVs are sorted as per their similarity matrix. Then the CVs with the heighest similarity is considered. 5 top CVs are selected for each company but the parameter can be changed.
    """

    def __init__(self, k):
        self.number_of_top_candidates = k

    def load_job_description_data(self, json_file_path):
        # Load your CV data from cv_data.json
        data = {}
        with open(f"{json_file_path}", "r") as file:
            # Load json data from the file.
            data = json.load(file)
            # Close the file.
            file.close()
        return data

    def find_the_key(self, company_and_job_descriptions, job_description):
        # Function to find the key (company name) for a given job description
        for key, value in company_and_job_descriptions.items():
            if value == job_description:
                return key

    def company_category_matches(self):
        """Find The Domain For Which Company Is Looking For

        # A slight varition is done while implementing the solution. Instead of iterating all the CVs for finding the similarity it is better to first determine the category or the domain for which the company is lookin for. For ecample if the company is looking for Web developer then there is no point is processing the CVs that belong to teaching. The idea is to first determine the domain which best suits the role and later find the similarity matrix from only the relevant CVs. This helps to speed up the process by ~5 times as well as improve the accuracy of the algorithm being used.

        # The data is stored in json format in the matched.json file. The file contains the company name along with the role for which they are looking for.

        # The idea is to find the similarity matrix between the 'job_title' field and the 'Category'. Next the pairs having the heighest similarity score is considered and **only those CVs are selected which belongs to the category later.**
        # **Note**: This is not 100% accurate."""

        # Free GPU memory
        _ = DeviceInteractions.empty_graphics_card()

        data = self.load_job_description_data("job_descriptions/cv_data.json")

        # Extract company names and job descriptions. This will hold the company name along with the job description.
        company_and_job_descriptions = {}
        for item in data:
            company_and_job_descriptions[item["company_name"]] = item["position_title"]

        # Load DistilBERT tokenizer and model on the GPU
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # Use the distilbert-base-uncased model
        model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
        cv_details = self.load_job_description_data("extracted/extracted_details.json")

        # Create a list of job descriptions
        job_descriptions = list(company_and_job_descriptions.values())

        # Tokenize and embed job descriptions using the tokenizer, model and the list comprehension technique.
        job_desc_embeddings = [
            model(
                **tokenizer(
                    job_desc,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
            ).last_hidden_state.mean(dim=1)
            for job_desc in job_descriptions
        ]

        # Initialize a dictionary to store collected CVs for each job description
        _ = {job_desc: [] for job_desc in job_descriptions}

        categories = []
        for cv in cv_details:
            categories.append(cv["Category"])

        # Convert the list of categories to a set to get unique categories
        unique_categories = list(set(categories))

        # Print the unique categories
        print("The unique domains present are as follows: {}".format(unique_categories))

        store = {}

        # Create a dictionary to store the similarity scores for each CV
        similarities = {}

        # Iterate over company names and job descriptions
        for company_name, job_desc in company_and_job_descriptions.items():
            # Calculate embeddings for the job description
            job_desc_embedding = job_desc_embeddings[
                job_descriptions.index(job_desc)
            ].squeeze(0)

            # Iterate over CVs
            for cv in unique_categories:
                cv_text = f"{cv}"

                # Create embeddings for the CV
                cv_embedding = model(
                    **tokenizer(
                        cv_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(device)
                ).last_hidden_state.mean(dim=1)
                cv_embedding = cv_embedding.squeeze(0)

                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    job_desc_embedding, cv_embedding, dim=0
                ).item()

                # Store the similarity score in the dictionary
                if company_name not in similarities:
                    similarities[company_name] = {}
                similarities[company_name][cv] = similarity

        # Find the top CV for each job description
        top_matching_cv = {}
        for company_name, similarity_scores in similarities.items():
            top_cv = max(similarity_scores, key=similarity_scores.get)
            top_matching_cv[company_name] = (top_cv, similarity_scores[top_cv])

        # Print the top matching CV for each job description
        for company_name, (top_cv, similarity) in top_matching_cv.items():
            # print(f"Company: {company_name}")
            # print(f"Top Matching CV: {top_cv}")
            # print(f"Similarity Score: {similarity}")
            store[company_name] = top_cv

        print("Companies and the domain they are looking for: {}".format(store))

        with open("matches/matched.json", "w") as file:
            json.dump(store, file)
            file.close()

    def domain_preference(self, company_name):
        comapany_preferred_domain = {}
        domain = {}
        with open("matches/matched.json", "r") as file:
            comapany_preferred_domain = json.load(file)
            file.close()
        if company_name in comapany_preferred_domain:
            print(
                f"Company name: {company_name}      Hiring in the domain of the following: {comapany_preferred_domain[company_name]}"
            )
            domain = comapany_preferred_domain[company_name]

        return domain

    def recommend_top_k_cvs(self, collected_cvs):
        # Initialize a dictionary to store top 5 CVs for each job description
        top_k_cvs = {}
        # Sort the collected CVs by similarity score and select the top 5
        for job_desc, cvs in collected_cvs.items():
            top_k_cvs[job_desc] = sorted(cvs, key=lambda x: x[1], reverse=True)[
                : self.number_of_top_candidates
            ]

        return top_k_cvs

    def shortlisted_candidates(self, company_and_job_descriptions, top_k_cvs):
        print(company_and_job_descriptions)
        shortlisted_cvs = {}
        for job_desc, cvs in top_k_cvs.items():
            # Call the function to find the company name corresponding to the job description.
            company = self.find_the_key(company_and_job_descriptions, job_desc)
            print(f"Top 5 CVs for '{company}':")

            # List to store all the selected resumes.
            list_of_selected_resumes = []
            for cv, similarity in cvs:
                print(f"CV: {cv}, Similarity Score: {similarity}")
                list_of_selected_resumes.append(cv)
            shortlisted_cvs[company] = list_of_selected_resumes
            print()

        return shortlisted_cvs

    def return_selected_cvs(self, category):
        selected_cvs = []  # Create an empty list to store selected CV details

        # Load job descriptions and extracted CV details from JSON
        with open("extracted/extracted_details.json", "r") as json_file:
            cv_details = json.load(json_file)

            # Iterate over each CV detail
            for cv_detail in cv_details:
                # Check if the "Category" key has the value of the passed category
                if cv_detail.get("Category") == category:
                    selected_cvs.append(
                        cv_detail
                    )  # Add the CV detail to the selected_cvs list

        return selected_cvs

    def similarity_matrix(self):
        _ = DeviceInteractions.empty_graphics_card()
        data = self.load_job_description_data("job_descriptions/cv_data.json")

        # Extract company names and job descriptions. This will hold the company name along with the job description.
        company_and_job_descriptions = {}
        for item in data:
            company_and_job_descriptions[item["company_name"]] = item["model_response"]

        # Load DistilBERT tokenizer and model on the GPU
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # Use the distilbert-base-uncased model
        model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

        # Load job descriptions and extracted CV details from JSON
        with open("extracted/extracted_details.json", "r") as json_file:
            # Load the json file.
            cv_details = json.load(json_file)
            # Close the file.
            json_file.close()

        # Create a list of job descriptions
        job_descriptions = list(company_and_job_descriptions.values())

        # Tokenize and embed job descriptions using the tokenizer, model and the list comprehension technique.
        job_desc_embeddings = [
            model(
                **tokenizer(
                    job_desc,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
            ).last_hidden_state.mean(dim=1)
            for job_desc in job_descriptions
        ]

        # Initialize a dictionary to store collected CVs for each job description
        collected_cvs = {job_desc: [] for job_desc in job_descriptions}

        # Tokenize and embed CV details
        for company_name, job_desc in company_and_job_descriptions.items():
            # Program to extract the directory to choose

            domain = self.domain_preference(company_name)

            cv_details = self.return_selected_cvs(domain)

            # Iterate over job descriptions
            for cv in cv_details:
                """Use the join method to add categoiry, skills, and education as text data. This will be the raw text to be tokenized representing most of the information of the user."""

                cv_text = f"{cv['Category']} {cv['Experience']} {', '.join(cv['Skills'])} {', '.join(cv['Education'])}"

                # Create embeddings of the raw text processed in the earlier step. The model is made to run on GPU if available.
                # Padding is set to true to match the text datas. truncation is set to True to ensuer that the texts lies within a max limit. Max length defines the maximum length of the texts.
                cv_embedding = model(
                    **tokenizer(
                        cv_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(device)
                ).last_hidden_state.mean(dim=1)

                # Calculate cosine similarity between job descriptions and CVs using PyTorch
                cv_embedding = cv_embedding.squeeze(0)  # Remove the batch dimension

                job_desc_embedding = job_desc_embeddings[
                    job_descriptions.index(job_desc)
                ].squeeze(
                    0
                )  # Get the corresponding job description embedding

                # Perform similarity operation using the cosine similary of PyTorch. Note that I used this instead of the sklearn implementation to run the operation of GPU devices. This will help to perform the operation much faster.
                similarity = torch.nn.functional.cosine_similarity(
                    job_desc_embedding, cv_embedding, dim=0
                ).item()

                # Store the CV and similarity score
                collected_cvs[job_desc].append((cv["PDFFilename"], similarity))
        return (company_and_job_descriptions, collected_cvs)

    def save_shortlisted_cvs(self, shortlisted_cvs):
        # Save the cvs of the final shortlisted candidates
        with open("shortlisted/shortlisted_cvs.json", "w") as file:
            json.dump(shortlisted_cvs, file, indent=4)
            # Close the file.
            file.close()


def main():
    # Specify the fields you want to include in the horizontal format

    company_fields = CompanyFields()

    _ = company_fields.get_company_fields()

    # Transform the data to horizontal format
    _ = company_fields.convert_to_json_format()

    _ = company_fields.save_job_description_data()

    # Perform data extraction from the pdf.
    PATH = "archive/data/data"

    cv_details = CvDataExtraction(PATH)

    cv_details.extract_details_from_cv()

    # The following is the code to produce the json file containing the comapany and the category of the role they are looking for.

    company_category = CvSeclection(5)

    _ = company_category.company_category_matches()
    # Tokenize and preprocessing along with the similarity matrix.

    company_and_job_descriptions, collected_cvs = company_category.similarity_matrix()

    # Call the function to get the top k cvs which matches with the job description.

    top_k_cvs = company_category.recommend_top_k_cvs(collected_cvs)

    # Print the top 5 CVs for each job description

    shortlisted_cvs = company_category.shortlisted_candidates(
        company_and_job_descriptions, top_k_cvs
    )

    # Call the function to save the shortlisted cvs.

    company_category.save_shortlisted_cvs(shortlisted_cvs)


# Call the driver code of tha program.

if __name__ == "__main__":
    main()
