"""Script that predicts the corresponding LISA sheets of the given questions"""
import os
import time
from random import shuffle, seed
import glob
import csv
import json
import re
from collections import defaultdict
import pandas as pd
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup
from groq import Groq

count = 1
seed(42)
# I've put multiple api keys in order to have the ability to swap between them without stopping the script so as not to go over the GroqCloud free token limit
clients = [Groq(api_key="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"), Groq(api_key="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")] # TODO remove
db_params = {
    'dbname': 'uness',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5432',
}
# Using SQLAlchemy engine (to fix a warning)
engine = create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

csv_file_path = 'predictions.csv'

try:
    if os.path.getsize(csv_file_path) > 0:
        existing_df = pd.read_csv(csv_file_path)
        if not existing_df.empty:
            last_question_id = existing_df['questionID'].max()
        else:
            last_question_id = 0  # If the file is empty
    else:
        last_question_id = 0  # If the file is empty
except FileNotFoundError:
    last_question_id = 0  # If the file does not exist

questions_query = text("""SELECT q.id as question_id, q.questiontext, m.metadata_id FROM mdl_question q JOIN mdl_uness_question_metadata m ON q.id = m.question_id""")
meta_query = text("""SELECT id as metadata_id, label, description FROM mdl_uness_metadata WHERE type = 'learning_objective' and label like 'learning_obj%'""")

connection = engine.connect()
questions_df = pd.read_sql_query(questions_query, connection)
meta_df = pd.read_sql_query(meta_query, connection)

# Merging our data
merged_df = questions_df.merge(meta_df, on='metadata_id', how='inner')
filtered_df = merged_df[merged_df['question_id'] > last_question_id] # to work only the questions which were not already predicted
nb_questions = 200  # If you want to run the script for each 200 questions and double check after each batch
ordered_df = filtered_df.sort_values(by='question_id').head(nb_questions)

metaItems = pd.read_csv('results-withCorrections.csv', delimiter=';')


def clean_up(html):
    """Helps with cleaning up questions (sometimes they have HTML code in them)"""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def get_correct_answer(question_id):
    """Returns one of the correct answers so that we can put it in the context"""
    answer_query = text(f"SELECT answer FROM public.mdl_question_answers WHERE question = {question_id} AND fraction > 0")
    answer_df = pd.read_sql_query(answer_query, connection)
    result = "Oops couldn't find a correct answer"
    if not answer_df.empty:
        correct_answers = "sep%".join(clean_up(answer) for answer in answer_df['answer'])
        result = correct_answers
    return result


ordered_df['correct_answer'] = ordered_df['question_id'].apply(lambda qid: get_correct_answer(qid))

def get_lisa_item(id, items):
    """Get the corresponding LISA item"""
    return items.loc[id]

def summarize_text(text, max_length=50):
    """Limits the length of a certain text"""
    return text[:max_length] + "..." if len(text) > max_length else text


def generate_prediction(context, question, correct_answer, client, chosenModel):
    """Helper func that generates a prediction using the context provided"""
    global count
    input_message = (
        f"Let's think step-by-step as a medical expert to identify the most relevant LISA sheets that have the answer to the given question. "
        f"First, we will carefully analyze the question, then we will review the possible LISA sheets, and finally, we will identify exactly five distinct sheets that are most likely to contain the correct answer.\n"
        f"Question: {question}\n"
        f"The correct answer: {correct_answer}\n"
        f"Possible LISA Sheets:\n{context}\n"
        f"Put your whole reply with ALL the steps and the result between <REPLY> and </REPLY> like this: <REPLY> [your reply] </REPLY>\n"
        f"Step 1: Analyze the key terms and understand the concepts in the question.\n"
        f"Then output your analysis during the 1st step between <STEP1> and </STEP1> like this: <STEP1> [your step 1 analysis] </STEP1>\n"
        f"Step 2: Match these key terms with the title of each one of the LISA sheets and remember how close each sheet is to the question.\n"
        f"Then output your analysis during the 2nd step between <STEP2> and </STEP2> like this: <STEP2> [your step 2 analysis] </STEP2>\n"
        f"Step 3: Identify the top five LISA sheets that most closely match the key terms and concepts from the question. Even if you think there might be other close ones other than the ones provided, just choose the closest ones out of the ones you have.\n"
        f"Then output your analysis during the 3rd step between <STEP3> and </STEP3> like this: <STEP3> [your step 3 analysis] </STEP3>\n"
        f"After all the steps, output the chosen LISA sheet IDs between <RESULT> and </RESULT> like this: <RESULT>OIC-259-01-A, OIC-022-06-B, OIC-063-13-A, OIC-361-16-A, OIC-111-02-B, OIC-093-01-A, OIC-330-02-A</RESULT>. No explanation or anything else.\n"
        f"REMEMBER to CLOSE EVERY markup tag you open when you are generating the answer and to always go through all the steps and give a result."
    )
    
    with open('sheet_prompts.txt', 'a') as prompt_file:
        prompt_file.write(f"{count}: " + input_message + '\n')
        
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model=chosenModel,
        temperature=0.0,
        top_p=1.0
    )

    result = chat_completion.choices[0].message.content
    
    with open('sheet_outputs.txt', 'a') as prompt_file:
        prompt_file.write(f"\n{count}: {question} -> Output:\n{result}\n")
    
    # Extract the predicted sheets using the markers
    start_marker = "<RESULT>"
    # end_marker = "</RESULT>"
    
    start_index = result.find(start_marker) + len(start_marker)
    
    if start_index != -1:
        predicted_sheets = result[start_index:].strip()
        # Extract the sheet identifiers using regex
        sheet_identifiers = re.findall(r'OIC-\d{3}-\d{2}-[AB]', predicted_sheets)
    else:
        print("Failed to extract sheets")
        sheet_identifiers = ["OIC-001-01-A"]
    
    with open('sheet_results.txt', 'a') as prompt_file:
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
    summarized_body = summarize_text(clean_up(body), max_length=1700) # Summarize the body
    parsed_data = ' '.join(relevant_fields)
    return parsed_data, summarized_body

def full_text_prediction(context, sheets, question, correct_answer, client, chosenModel):
    """Final prediction with 3 'full' LISA sheets"""
    global count
    index = 1
    shuffle(sheets)
    
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
        f"Let's think step-by-step to identify as a medical expert the most relevant LISA sheet that contains the information which will guide a student to the correct answer to the given question. "
        f"First, we will carefully analyze the question and the provided correct answer. "
        f"Then, we will review the content of the top five LISA sheets, and finally, we will identify the single most relevant LISA sheet.\n"
        f"Question: {question}\n"
        f"The correct answer: {correct_answer}\n"
        f"Possible LISA Sheets:\n{context}\n"
        f"Put your whole reply with ALL the steps and the result between <REPLY> and </REPLY> like this: <REPLY> [your reply] </REPLY>\n"
        f"Step 1: Analyze the key terms and concepts in the question.\n"
        f"Then output your analysis during the 1st step between <STEP1> and </STEP1> like this: <STEP1> [your step 1 analysis] </STEP1>\n"
        f"Step 2: Match these key terms with the descriptions and content provided in the following LISA sheets.\n"
        f"Then output your analysis during the 2nd step between <STEP2> and </STEP2> like this: <STEP2> [your step 2 analysis] </STEP2>\n"
        f"Step 3: Identify the LISA sheet that most closely matches the key terms and concepts from the question and contains the correct answer. "
        f"Even if you think there might be other close ones other than the ones provided, just choose the closest one out of the ones you have.\n"
        f"Then output your analysis during the 3rd step between <STEP3> and </STEP3> like this: <STEP3> [your step 3 analysis] </STEP3>\n"
        f"After all the steps, output the chosen LISA sheet ID between <RESULT> and </RESULT> like this: <RESULT>OIC-001-01-A</RESULT>. No explanation or anything else.\n"
        f"REMEMBER to CLOSE EVERY markup tag you open when you are generating the answer and to always go through all the steps and give a result."
    )

    with open('sheet_prompts.txt', 'a') as prompt_file:
        prompt_file.write(f"{count}: " + input_message_final + '\n-----\n')

    chat_completion_final = client.chat.completions.create(
        messages=[{"role": "user", "content": input_message_final}],
        model=chosenModel,
        temperature=0.0,
        top_p=1.0
    )
    result = chat_completion_final.choices[0].message.content


    with open('sheet_outputs.txt', 'a') as prompt_file:
        prompt_file.write(f"\n{count}: FINAL Output  : {result}\n")

    # Extract the predicted sheet using the markers
    start_marker = "<RESULT>"

    start_index = result.find(start_marker) + len(start_marker)

    if start_index != -1:
        predicted_sheet = result[start_index:].strip()
        # Extract the sheet identifier using regular expressions
        sheet_identifier = re.findall(r'OIC-\d{3}-\d{2}-[AB]', predicted_sheet)
        if sheet_identifier:
            final_sheet_id = sheet_identifier[0]
        else:
            print("\nFailed to extract a valid sheet ID. Using random.\n")
            final_sheet_id = sheets[0]
    else:
        print("\nFailed to extract a valid sheet ID. Using random.\n")
        final_sheet_id = sheets[0]

    with open('sheet_results.txt', 'a') as prompt_file:
        prompt_file.write(f"{count}: " + final_sheet_id + '\n-----\n')

    count += 1
    print(f"API Final Sheet ID : {final_sheet_id}")
    return final_sheet_id

def predict_lisa_sheet(question, correct_answer, item, correct_lisa_sheet, max_tokens, isTest, client, chosenModel):
    """Predicts the corresponding LISA sheet for a given question"""
    context = ""
    txt_files = glob.glob(os.path.join(f"lisa_sheets/IC-{item}/", '*.txt'))
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            content, _ = parse_lisa_sheet(f.read())
            context += f"\n{content};\n"
    predicted_sheets = generate_prediction(context, question, correct_answer, client, chosenModel)
    
    full_text_context = ""
    predicted_sheet = full_text_prediction(full_text_context, predicted_sheets, question, correct_answer, client, chosenModel)
    try:
        predicted_rank = predicted_sheet.split('-')[3]

    except Exception: # TODO this shouldn't happen I'm pretty sure, check later to remove it if dead code
        predicted_sheet = predicted_sheets[0]
        predicted_rank = predicted_sheet.split('-')[3]
        
    return predicted_sheet, predicted_rank

def getItem(metaLabel):
    """Get the corresponding item from the csv file"""
    row = metaItems[metaItems['learning_objective'] == metaLabel]
    if not row.empty:
        item = int(row['predictedItem'].values[0])
        formatted_item = str(item).zfill(3) # Item numbers are always 3 characters long (eg: 003)
        return formatted_item
    else:
        return '000'

def match_questions_to_sheets(questions, max_tokens, isTest, chosenModel):
    """Matches a LISA sheet for each question of a df and optionally stores the results in a CSV file."""
    results = []
    step = 1  # Represents which question we are on
    turn = 0  # The script will keep swapping between the clients every 35 questions in order to never go over the token limit of GroqCloud
    client = clients[turn]

    csv_file_path = 'predictions.csv'
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as csv_file:
        fieldnames = ['questionID', 'lisa_sheet_ID', 'rank']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if (not file_exists or os.path.getsize(csv_file_path) == 0) and not isTest:  # Only write header if the file is new and if not testing
            writer.writeheader()
            csv_file.flush()

        for _, row in questions.iterrows():
            if step % 35 == 0:
                turn = (turn + 1) % 2
                client = clients[turn]

            question_id = row['question_id']
            question = row['questiontext']
            correct_answers = row['correct_answer']
            print("-> QUESTION : " + f"{step} - ID: {question_id}")
            step += 1
            votes = defaultdict(int)  # A Map in which the default value is 0 (because every ID has 0 votes at first)

            for correct_answer in correct_answers.split("sep%"):
                retry_count = 0
                max_retries = 3  # Limit the number of retries to 3
                while retry_count < max_retries:
                    try:
                        if isTest:
                            correct_lisa_sheet = row['lisa_sheet_id']
                            predicted_item = correct_lisa_sheet[4:7]
                        else:
                            metaLabel = row['label']
                            predicted_item = getItem(metaLabel)
                            correct_lisa_sheet = 'OIC'
                        if predicted_item == '000':
                            raise ValueError("Couldn't predict an item.")

                        predicted_sheet, _ = predict_lisa_sheet(
                            question, correct_answer, predicted_item, correct_lisa_sheet,
                            max_tokens, isTest, client, chosenModel
                        )
                        break  # Break out of the retry loop if success

                    except Exception as e:
                        if "Error code: 503" in str(e):
                            print("503 Service Unavailable. Retrying in 10 seconds...")
                            retry_count += 1
                            time.sleep(10)  # Wait for 10 seconds before retrying
                        else:
                            predicted_sheet = 'OIC-001-01-A'
                            break  # Exit the retry loop on other exceptions

                if retry_count == max_retries:
                    print("Max retries reached. Assigning default value.")
                    predicted_sheet = 'OIC-001-01-A'

                votes[predicted_sheet] += 1

            votes['OIC-001-01-A'] = 0.1  # To not choose it if other sheets have been predicted only once
            final_predicted_sheet = max(votes, key=votes.get)  # Most voted sheet
            final_predicted_rank = final_predicted_sheet[-1]
            print(f"\n----> The votes are in. The chosen sheet is: {final_predicted_sheet}\n")

            results.append({
                'question_id': row['question_id'],
                'predicted_sheet_id': final_predicted_sheet,
                'predicted_rank': final_predicted_rank,
            })

            if not isTest:
                writer.writerow({
                    'questionID': row['question_id'],
                    'lisa_sheet_ID': final_predicted_sheet,
                    'rank': final_predicted_rank
                })
                csv_file.flush()  # Ensure the data is written to the file immediately

            with open('sheet_results.txt', 'a') as prompt_file:
                prompt_file.write(f"{step}: FINAL PRED -> {final_predicted_sheet}\n-----\n")

    return pd.DataFrame(results)  # This is used for the Validation (accuracies are computed from this df)




if __name__ == "__main__":
    max_tokens = 8000
    isTest = False
    chosenModel = "llama-3.1-70b-versatile"
    print(ordered_df)
    match_questions_to_sheets(ordered_df, max_tokens, isTest, chosenModel)