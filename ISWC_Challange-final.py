import json

# ==============================================================================
# Preparing the Dataset - WordNet
# ==============================================================================

# Read the input JSON file
with open('wordnet_train.json', 'r') as file:
    input_data = json.load(file)

output_data = []

# Process each item in the input JSON
for item in input_data:
    term = item["term"]
    pos_type = item["type"]
    sentence = item["sentence"]
    
    if sentence:
        user_content = f"Perform a sentence completion on the following sentence: The part of speech of the term \"{term}\" in the sentence \"{sentence}\" is ___ "
        assistant_content = f"The part of speech of the term \"{term}\" in the sentence \"{sentence}\" is {pos_type}."
    else:
        user_content = f"Perform a sentence completion on the following sentence: The part of speech of the term \"{term}\" is ___ "
        assistant_content = f"The part of speech of the term \"{term}\" is {pos_type}."
    
    output_data.append({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    })

# Write the output data to a JSONL file
with open('wordnet_train.jsonl', 'w') as file:
    for entry in output_data:
        file.write(json.dumps(entry) + "\n")

print("The JSON file has been successfully converted to JSONL format.")

# ==============================================================================
# Preparing the Dataset - GeoNames
# ==============================================================================

# Read the input JSON file
with open('geonames_train.json', 'r') as file:
    input_data = json.load(file)

output_data = []

# Process each item in the input JSON
for item in input_data:
    term = item["term"]
    geo_type = item["type"]

    user_content = f"Perform a sentence completion on the following sentence: \"{term}\" geographically is a ___ "
    assistant_content = f"\"{term}\" geographically is a {geo_type}."
    
    output_data.append({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    })

# Write the output data to a JSONL file
with open('geonames_train.jsonl', 'w') as file:
    for entry in output_data:
        file.write(json.dumps(entry) + "\n")

print("The JSON file has been successfully converted to JSONL format.")

# ==============================================================================
# Preparing the Dataset - UMLS MedCin
# ==============================================================================

# Read the input JSON file
with open('medcin_train.json', 'r') as file:
    input_data = json.load(file)

output_data = []

# Process each item in the input JSON
for item in input_data:
    term = item["term"]
    umls_type = item["type"]
    
    user_content = f"Perform a sentence completion on the following sentence: \"{term}\" in medicine can be described as ___"
    assistant_content = f"The type of \"{term}\" in medicine can be described as: {umls_type}."
    
    output_data.append({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    })

# Write the output data to a JSONL file
with open('medcin_train.jsonl', 'w') as file:
    for entry in output_data:
        file.write(json.dumps(entry) + "\n")

print("The JSON file has been successfully converted to JSONL format.")

# ==============================================================================
# Preparing the Dataset - UMLS NCI
# ==============================================================================

# Read the input JSON file
with open('nci_train.json', 'r') as file:
    input_data = json.load(file)

output_data = []

# Process each item in the input JSON
for item in input_data:
    term = item["term"]
    umls_type = item["type"]
    
    user_content = f"Perform a sentence completion on the following sentence: \"{term}\" in medicine can be described as ___"
    assistant_content = f"The type of \"{term}\" in medicine can be described as: {umls_type}."
    
    output_data.append({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    })

# Write the output data to a JSONL file
with open('nci_train.jsonl', 'w') as file:
    for entry in output_data:
        file.write(json.dumps(entry) + "\n")

print("The JSON file has been successfully converted to JSONL format.")

# ==============================================================================
# Preparing the Dataset - UMLS SNOMEDCT_US
# ==============================================================================

# Read the input JSON file
with open('snomedct_us_train.json', 'r') as file:
    input_data = json.load(file)

output_data = []

# Process each item in the input JSON
for item in input_data:
    term = item["term"]
    umls_type = item["type"]
    
    user_content = f"Perform a sentence completion on the following sentence: \"{term}\" in medicine can be described as ___"
    assistant_content = f"The type of \"{term}\" in medicine can be described as: {umls_type}."
    
    output_data.append({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    })

# Write the output data to a JSONL file
with open('snomedct_us_train.jsonl', 'w') as file:
    for entry in output_data:
        file.write(json.dumps(entry) + "\n")

print("The JSON file has been successfully converted to JSONL format.")

# ==============================================================================
# OpenAI API Setup and Installation
# ==============================================================================

# Install OpenAI library
# pip install openai --user
# pip install openai --upgrade

import openai
print(openai.VERSION)

# ==============================================================================
# Uploading the Training Dataset for Fine-Tuning
# ==============================================================================

# Define your OpenAI API key
from openai import OpenAI
client = OpenAI(api_key='your OpenAI key')

def upload_file_for_fine_tuning(file_path):
    try:
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose="fine-tune"
            )
        print("File uploaded successfully:", response)
    except Exception as e:
        print("An error occurred:", e)

# Example usage
upload_file_for_fine_tuning("wordnet_train.jsonl")

# ==============================================================================
# Creating the Fine-Tuning Job
# ==============================================================================

def create_fine_tuning_job(training_file_id, model="gpt-3.5-turbo-0125", suffix="Biological_p_all"):
    try:
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model,
            suffix=suffix
        )
        print("Fine-tuning job created successfully:", response)
    except Exception as e:
        print("An error occurred:", e)

# Example usage
training_file_id = "file-tAzN3X2PJfZk1RHJdDlYYBcd"  # replace with your training file id
create
