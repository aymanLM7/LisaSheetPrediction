"""Script that predicts the corresponding LISA sheets of the given questions"""
import os
from random import shuffle, seed
import glob
import csv
import json
import re
from collections import defaultdict
import pandas as pd
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from gradio_client import Client

# model = "tinyllama" # uncomment this line if you want to use the tinyllama model
model = "phi3mini"
count = 1
seed(42)
db_params = {
    'dbname': 'uness',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5432',
}
# Using SQLAlchemy engine (to fix a warning)
engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

questions_query = text("""SELECT q.id as question_id, q.questiontext, m.metadata_id FROM mdl_question q JOIN mdl_uness_question_metadata m ON q.id = m.question_id""")
meta_query = text("""SELECT id as metadata_id, label, description FROM mdl_uness_metadata WHERE type = 'learning_objective' and label like 'learning_obj%'""")

connection = engine.connect()
questions_df = pd.read_sql_query(questions_query, connection)
meta_df = pd.read_sql_query(meta_query, connection)

# Merging our data
merged_df = questions_df.merge(meta_df, on='metadata_id', how='inner')
sampled_merged_df = merged_df.sample(n=5, random_state=22)

metaItems = pd.read_csv('results-withCorrections.csv', delimiter=';')

def clean_up(html):
    """Helps with cleaning up questions (sometimes they have HTML code in them)"""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def get_correct_answer(question_id): # TODO put it back (needs to be adapted for generated questions)
    """Returns one of the correct answers so that we can put it in the context"""
    answer_query = text(f"SELECT answer FROM public.mdl_question_answers WHERE question = {question_id} AND fraction > 0")
    answer_df = pd.read_sql_query(answer_query, connection)
    result = "Oops couldn't find a correct answer"
    if not answer_df.empty:
        correct_answers = "sep%".join(clean_up(answer) for answer in answer_df['answer'])
        result = correct_answers
    return result

sampled_merged_df['correct_answer'] = sampled_merged_df['question_id'].apply(lambda qid: get_correct_answer(qid))

def get_lisa_item(id, items):
    """Get the corresponding LISA item"""
    return items.loc[id]

def summarize_text(text, max_length=50):
    """Limits the length of a certain text"""
    return text[:max_length] + "..." if len(text) > max_length else text

def prepare_context(items):
    """Helper func that prepares context"""
    context = ""
    for _, item in items.iterrows():
        context += f"Item Number {item['Item']}: {item['Intitulé']};\n"
    return context


def generate_prediction(context, question, correct_answer):
    """Helper func that generates a prediction using the context with retry mechanism"""
    if model == "phi3mini":
        client = Client("eswardivi/Phi-3-mini-128k-instruct")
    elif model == "tinyllama":
        client = Client("TinyLlama/tinyllama-chat")
    elif model == "phi3small":
        client = Client("seyf1elislam/Phi-3-small-8k-instruct-7b")
    else:
        raise ValueError("The chosen model hasn't been implemented yet.")
    global count
    input_message = (
        f"Forget previous instructions. Reply only with: {{RESULT}}[ID 1], [ID 2], [ID 3]{{/RESULT}}. [ID] and identifiant represent the same thing so you have to choose from the provided IDs and give no explanation.\n"
        f"Provide the IDs of the 3 closest LISA sheets to help a student answer correctly (do NOT give less than 3 IDs).\n"
        f"Question: {question}\n"
        f"Answer: {correct_answer}\n"
        f"Possible LISA Sheets:\n{context}\n"
        f"Output only the chosen LISA sheet IDs between {{RESULT}} and {{/RESULT}} (write the {{RESULT}} so I can extract the result).\n"
    )
    with open(f"{model}_prompts.txt", 'a') as prompt_file:
        prompt_file.write(f"{count}: " + input_message + '\n')
        
    result = client.predict( # Check if these are the best params
		input_message,
		0,
		True,
        4096,
		api_name="/chat"
    )
    
    with open(f"{model}_outputs.txt", 'a') as prompt_file:
        prompt_file.write(f"\n{count}: {question} -> Output:\n{result}\n")
    
    # Extract the predicted sheets using the markers
    start = "{RESULT}"
    # end = "{/RESULT}"
    
    start_index = result.find(start) + len(start)
    
    if start_index != -1:
        predicted_sheets = result[start_index:].strip()
        # Extract the sheet identifiers using regex
        sheet_identifiers = re.findall(r'OIC-\d{3}-\d{2}-[AB]', predicted_sheets)
    else:
        print("Failed to extract sheets")
        sheet_identifiers = ["OIC-001-01-A"]
    
    with open(f"{model}_results.txt", 'a') as prompt_file:
        prompt_file.write(f"{count}: " + " / ".join(sheet_identifiers) + '\n-----\n')
        
    count += 1
    return sheet_identifiers


def parse_lisa_sheet(content):
    """Parses the LISA sheet to extract relevant fields"""
    relevant_fields = [
        'ID = ' + content.split('Identifiant=')[1].split('|')[0].strip(),  # Identifiant
        'Title = ' + content.split('Intitulé=')[1].split('|')[0].strip(),  # Intitulé
        'Description = ' + content.split('Description=')[1].split('|')[0].strip()   # Description
    ]
    body = content.split('}}')[-1].strip()

    if model == "phi3mini":
        max = 1500
    elif model == "tinyllama":
        max = 800
    elif model == "phi3small":
        max = 2500
    else:
        max = 500
    summarized_body = summarize_text(clean_up(body), max_length=max) # Summarize the body
    parsed_data = ' '.join(relevant_fields)
    return parsed_data, summarized_body

def full_text_prediction(context, sheets, question, correct_answer):
    """Final prediction with 3 'full' LISA sheets"""
    global count
    index = 1
    shuffle(sheets)
    if model == "phi3mini":
        client = Client("eswardivi/Phi-3-mini-128k-instruct")
    elif model == "tinyllama":
        client = Client("TinyLlama/tinyllama-chat")
    elif model == "phi3small":
        client = Client("seyf1elislam/Phi-3-small-8k-instruct-7b")
    else:
        raise ValueError("The chosen model hasn't been implemented yet.")
    
    for sheet in sheets:
        try:
            with open(f"lisa_sheets/{sheet[1:7]}/{sheet}.txt", 'r') as f:
                lisa_header, lisa_body = parse_lisa_sheet(f.read())
                context += f"LISA sheet {sheet}:\n" + lisa_header + "\n Content: " + lisa_body + ";\n\n"
            index += 1
        except Exception as e:
            print(f"Failed to read sheet {sheet}: {e}. Skipping...\n")
            continue
    
    input_message_final = (
        f"Forget previous instructions. Reply only with: {{RESULT}}[ID]{{/RESULT}}. [ID] and identifiant represent the same thing so you have to choose from the provided IDs and give no explanation.\n"
        f"Provide the ID of the closest LISA sheet to help a student answer correctly.\n"
        f"Question: {question}\n"
        f"Answer: {correct_answer}\n"
        f"Possible LISA Sheets:\n{context}\n"
        f"Output only the chosen LISA sheet ID between {{RESULT}} and {{/RESULT}} (write the {{RESULT}} so I can extract the result).\n"
    )
    with open(f"{model}_prompts.txt", 'a') as prompt_file:
        prompt_file.write(f"{count}: " + input_message_final + '\n-----\n')

    result = client.predict( # TODO: Check if these are the best params and if this works for all models or need to have different prediction for each one
		input_message_final,
		0,
		True,
        4096,
		api_name="/chat"
    )


    with open(f"{model}_outputs.txt", 'a') as prompt_file:
        prompt_file.write(f"\n{count}: FINAL Output  : {result}\n")

    # Extract the predicted sheet using the markers
    start = "{RESULT}"
    # end = "{/RESULT}"

    start_index = result.find(start) + len(start)

    if start_index != -1:
        predicted_sheet = result[start_index:].strip()
        # Extract the sheet identifier using regex
        sheet_identifier = re.findall(r'OIC-\d{3}-\d{2}-[AB]', predicted_sheet)
        if sheet_identifier:
            final_sheet_id = sheet_identifier[0]
        else:
            print("\n1Failed to extract a valid sheet ID. Using random.\n")
            final_sheet_id = sheets[0]
    else:
        print(f'start index: {start_index}\n')
        print("\n2Failed to extract a valid sheet ID. Using random.\n")
        final_sheet_id = sheets[0]

    with open(f"{model}_results.txt", 'a') as prompt_file:
        prompt_file.write(f"{count}: " + final_sheet_id + '\n-----\n')

    count += 1
    print(f"API Final Sheet ID : {final_sheet_id}")
    return final_sheet_id

def predict_lisa_sheet(question, correct_answer, item, correct_lisa_sheet, max_tokens, isTest):
    """Predicts the corresponding LISA sheet for a given question"""
    
    def get_top_3_sheets(group, question, correct_answer, double):
        """Gets the top 3 sheets from a group using generate_prediction function"""
        context = ""
        for sheet in group:
            if double:
                sheet = f"lisa_sheets/IC-{item}/{sheet}"
            with open(sheet, 'r') as f:
                content, _ = parse_lisa_sheet(f.read())
                context += f"\n{content};\n"
        return generate_prediction(context, question, correct_answer)

    # Split the sheets into groups of 10
    txt_files = glob.glob(os.path.join(f"lisa_sheets/IC-{item}/", '*.txt'))
    groups = [txt_files[i:i+10] for i in range(0, len(txt_files), 10)]
    
    top_sheets_from_groups = []
    for group in groups:
        top_sheets_from_group = get_top_3_sheets(group, question, correct_answer, False)
        for top_sheet in top_sheets_from_group:
            top_sheet += ".txt"
            top_sheets_from_groups.append(top_sheet)

    if len(groups) > 1:
        final_top_sheets = get_top_3_sheets(top_sheets_from_groups, question, correct_answer, True)
    else:
        final_top_sheets = top_sheets_from_groups

    stepOneCorrect = 0
    if isTest:
        for sheet in final_top_sheets:
            if sheet == correct_lisa_sheet:
                stepOneCorrect += 1
                break
    full_text_context = ""
    predicted_sheet = full_text_prediction(full_text_context, final_top_sheets, question, correct_answer)
    try:
        predicted_rank = predicted_sheet.split('-')[3]
    except Exception: 
        predicted_sheet = final_top_sheets[0]
        predicted_rank = predicted_sheet.split('-')[3]

    return predicted_sheet, predicted_rank, stepOneCorrect

def getItem(metaLabel):
    """Get the corresponding item from the csv file"""
    row = metaItems[metaItems['learning_objective'] == metaLabel]
    if not row.empty:
        item = int(row['predictedItem'].values[0])
        formatted_item = str(item).zfill(3) # Item numbers are always 3 characters long (eg: 003)
        return formatted_item
    else:
        return '000'

def match_questions_to_sheets(questions, max_tokens, isTest):
    """Matches a LISA sheet for each question of a df"""
    results = []
    step = 1
    stepOneCorrect = 0
    for _, row in questions.iterrows():
        question_id = row['question_id']
        question = clean_up(row['questiontext'])
        correct_answers = row['correct_answer']
        print("-> QUESTION : " + f"{step} - ID: {question_id}")
        step += 1
        votes = defaultdict(int)
        for correct_answer in correct_answers.split("sep%"):
            try:
                if isTest:
                    correct_lisa_sheet = row['lisa_sheet_id']
                    predicted_item = correct_lisa_sheet[4:7]
                else:
                    metaLabel = row['label']
                    predicted_item = getItem(metaLabel)
                    correct_lisa_sheet = None # Useless because it's not a test
                if predicted_item == '000':
                    raise ValueError("Couldn't predict an item.")
                predicted_sheet, _, temp = predict_lisa_sheet(question, correct_answer, predicted_item, correct_lisa_sheet, max_tokens, isTest)
                stepOneCorrect += temp
            except Exception as e: # In order to treat unexpected outputs from the LLM
                print(e)
                predicted_sheet = 'OIC-001-01-A'
            votes[predicted_sheet] += 1
        votes['OIC-001-01-A'] = 0.1 # To not choose it if other sheets have been predicted only once
        final_predicted_sheet = max(votes, key=votes.get) # Most voted sheet
        final_predicted_rank = final_predicted_sheet[-1]
        print(f"\n----> The votes are in. The chosen sheet is: {final_predicted_sheet}\n")
        results.append({
            'question_id': row['question_id'],
            'predicted_sheet_id': final_predicted_sheet,
            'predicted_rank' : final_predicted_rank,
        })
        with open(f"{model}_results.txt", 'a') as prompt_file:
            prompt_file.write(f"{count}: FINAL PRED -> {final_predicted_sheet}\n-----\n")
    return pd.DataFrame(results), stepOneCorrect # FIXME : this needs to be stored somewhere