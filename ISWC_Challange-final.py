import json

#preapring the dataset
#wordnet

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

#preapring the dataset
#geonames

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

#preapring the dataset
#UMLS medcin

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

#preapring the dataset
#UMLS nci

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

#preapring the dataset
#UMLS snomedct_us

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


pip install openai --user

pip install openai --upgrade 


import openai
print(openai.VERSION)

#upload the training dataset for fine-tuning
import openai

# Define your OpenAI API key
from openai import OpenAI
client = OpenAI(api_key = 'your  OpenAI key')

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


#creating the fine-tuning job
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
training_file_id = "file-tAzN3X2PJfZk1RHJdDlYYBcd" # replace with your training file id
create_fine_tuning_job(training_file_id, model="gpt-3.5-turbo-0125", suffix="wordnet_all")  # Change model and suffix as needed

# Retrieve the state of a fine-tune
client.fine_tuning.jobs.retrieve("ftjob-13vPfnbqztHnpD9Awq8TT4xd") # replace with your job id


#evaluating results - wordnet
import openai

# Define your OpenAI API key
from openai import OpenAI
client = OpenAI(api_key = 'Your OpenAI key')

def get_term_types(term, sentence):
    if sentence:
        prompt = f"Perform a sentence completion on the following sentence: The part of speech of the term \"{term}\" in the sentence \"{sentence}\" is ___"
    else:
        prompt = f"Perform a sentence completion on the following sentence: The part of speech of the term \"{term}\" is ___"
    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:kansas-state-university:wordnet-all:9gNNcSRE",  # change to your finetuned model name
            messages=[{'role': 'user', 'content': prompt}],
            #max_tokens=50,
            #n=1,
            #stop=None,
            temperature=0.0,
        )
        completion_text = response.choices[0].message.content.strip()
        # Extract the types from the completion text
        if "is" in completion_text:
            types_start_index = completion_text.index("is") + len("is")
            types_text = completion_text[types_start_index:].strip()
            # Remove trailing period if present
            if types_text.endswith("."):
                types_text = types_text[:-1]
            types = [t.strip() for t in types_text.split(",")]
        else:
            types = []
        return types
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    results = []

    for entry in data:
        term = entry['term']
        sentence = entry['sentence']
        types = get_term_types(term, sentence)
        results.append({
            "ID": entry['ID'],
            "type": types
        })

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

# Example usage
input_file = 'A.1(FS)_WordNet_Test.json'  # Replace with your input file path
output_file = 'output_wordnet.json'  # Replace with your output file path
process_file(input_file, output_file)

#Cleaning up the output file - wordnet
import json

def extract_last_word_from_type(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    results = []

    for entry in data:
        types = entry['type']
        # Extract the last word from each type entry
        last_words = [t.split()[-1] for t in types]
        results.append({
            "ID": entry['ID'],
            "type": last_words
        })

    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

# Example usage
input_file = 'output_wordnet.json'  # Replace with your input file path
output_file = 'output_wordnet_new.json'  # Replace with your output file path
extract_last_word_from_type(input_file, output_file)

#evaluating results - geonames 
import openai
import json
from openai import OpenAI
client = OpenAI(api_key = 'Your OpenAI key')


def get_term_types(term):
    prompt = f"Perform a sentence completion on the following sentence: \"{term}\" geographically is a ___."
    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:kansas-state-university:geonames-10percent:9gnjksT9",  # change to your finetuned model name
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
        )
        completion_text = response.choices[0].message.content.strip()
        
        # Extract the words after "is a" from the completion text
        if "is a" in completion_text:
            types_text = completion_text.split("is a", 1)[1].strip()
            # Remove trailing period if present
            if types_text.endswith("."):
                types_text = types_text[:-1]
            types = [t.strip() for t in types_text.split(",")]
        else:
            types = []
        
        return types, completion_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def process_file(input_file, output_file, response_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    results = []
    responses = []

    for entry in data:
        term = entry['term']
        last_word, completion_text = get_term_types(term)
        if last_word and completion_text:
            result = {
                "ID": entry['ID'],
                "type": last_word
            }
            response = {
                "ID": entry['ID'],
                "response": completion_text
            }
            results.append(result)
            responses.append(response)

            # Write the results and responses to files as the code runs
            with open(output_file, 'w') as outfile:
                json.dump(results, outfile, indent=4)
            
            with open(response_file, 'w') as respfile:
                json.dump(responses, respfile, indent=4)

            # Print the extracted types for the term
            print(f"Extracted types for term '{term}': {result['type']}")

# Example usage
input_file = 'A.2(FS)_GeoNames_Test.json'  # Replace with your input file path
output_file = 'output_geonames.json'  # Replace with your output file path
response_file = 'response_geonames.json'  # Replace with your response file path
process_file(input_file, output_file, response_file)

#evaluating results - umls 
import openai
import json
from openai import OpenAI
client = OpenAI(api_key = 'your OpenAI key')


def get_term_types(term):
    prompt = f"Perform a sentence completion on the following sentence: \"{term}\" in medicine can be described as ___ ."
    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:kansas-state-university:nci-all:9lWu1IfQ",  # change to your finetuned model name
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
        )
        completion_text = response.choices[0].message.content.strip()
        
        # Extract the words after "as" from the completion text
        if "as" in completion_text:
            types_text = completion_text.split("as", 1)[1].strip()
            # Remove trailing period if present
            if types_text.endswith("."):
                types_text = types_text[:-1]
            types = [t.strip() for t in types_text.split(",")]
        else:
            types = []
        
        return types, completion_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def process_file(input_file, output_file, response_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    results = []
    responses = []

    for entry in data:
        term = entry['term']
        last_word, completion_text = get_term_types(term)
        if last_word and completion_text:
            result = {
                "ID": entry['ID'],
                "type": last_word
            }
            response = {
                "ID": entry['ID'],
                "response": completion_text
            }
            results.append(result)
            responses.append(response)

            # Write the results and responses to files as the code runs
            with open(output_file, 'w') as outfile:
                json.dump(results, outfile, indent=4)
            
            with open(response_file, 'w') as respfile:
                json.dump(responses, respfile, indent=4)

            # Print the extracted types for the term
            print(f"Extracted types for term '{term}': {result['type']}")

# Example usage
input_file = 'A.3(FS)_UMLS_NCI_Test.json'  # Replace with your input file path
output_file = 'output_nci.json'  # Replace with your output file path
response_file = 'response_nci.json'  # Replace with your response file path
process_file(input_file, output_file, response_file)

