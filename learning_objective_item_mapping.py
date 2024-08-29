"""Script that makes the learning_objective (metadata) to knowledge item mapping"""
from random import shuffle, seed
import csv
import json
import re
import pandas as pd
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup
from groq import Groq

count = 1
seed(42)
client = Groq(api_key= "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") # TODO remove
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

with open('lisa_sheets/ic.json', 'r') as file:
    lisa_items = json.load(file)

items_data = [
    {
        "Item": element["title"]["Item"],
        "Intitulé": element["title"]["Intitule"]
    }
    for element in lisa_items["cargoquery"]
]

items_df = pd.DataFrame(items_data)

def prepare_context(items):
    """Helper func that lists off all the potential knowledge items"""
    context = ""
    for _, item in items.iterrows():
        context += f"Item Number {item['Item']}: {item['Intitulé']};\n"
    return context

def clean_up(html):
    """Helps with cleaning up questions (most questions have HTML code in them)"""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def predict_items(context, meta_description, amount, label ,step):
    input_message = (
        f"Let's think step-by-step as a medical expert to identify the {amount} most relevant LISA items for the given description. You have to finish all the steps! First, we will carefully analyze the description, then we will review the possible LISA items, and finally, we will identify exactly the {amount} most likely items that the description is about.\n"
        f"Description of the item that we are looking for: {meta_description}\n"
        f"Possible LISA Items:\n{context}\n"
        f"Step 1: Analyze the key terms and understand the concepts in the description.\n"
        f"Step 2: Match these key terms with the title of each one of the LISA items, in most cases the closest item is going to have similar words that are in the description.\n"
        f"Step 3: Identify the best {amount} LISA items that most closely matches the key terms and concepts from the description.\n"
        f"After all the steps, output the chosen LISA items' Numbers between <RESULT> and </RESULT> like this: <RESULT>1, 2, 3</RESULT>. No explanation or anything else."
    )

    with open('prompts.txt', 'a') as prompt_file:
        prompt_file.write(f"\n{step}: {label} -> Prompt:\n{input_message}\n")

    chat_completion_final = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        temperature=0.1,
        top_p=0.9
    )

    result = chat_completion_final.choices[0].message.content

    start_marker = "<RESULT>"

    start_index = result.find(start_marker) + len(start_marker)

    if start_index != -1:
        items_string = result[start_index:].strip()
        predicted_items = [item.strip() for item in items_string.split(',')]

        with open('outputs.txt', 'a') as prompt_file:
            prompt_file.write(f"\n{step}: {label} -> Output:\n{result}\n")

    else:
        raise ValueError("Failed to extract items")
    print(f"\n {step}: {label} -> {predicted_items}\n")
    return predicted_items

def predict_final_item(context, meta_description, label, step):
    input_message = (
        f"Let's think step-by-step as a medical expert to identify the most relevant LISA item for the given description. You have to finish all the steps! First, we will carefully analyze the description, then we will review the possible LISA items, and finally, we will identify exactly the one most likely item that the description is about.\n"
        f"Description of the item that we are looking for: {meta_description}\n"
        f"Possible LISA Items:\n{context}\n"
        f"Step 1: Analyze the key terms and understand the concepts in the description.\n"
        f"Step 2: Match these key terms with the title of each one of the LISA items, in most cases the closest item is going to have similar words that are in the description.\n"
        f"Step 3: Identify the best LISA item that most closely matches the key terms and concepts from the description.\n"
        f"After all the steps, output the chosen LISA item's Number between <RESULT> and </RESULT> like this: <RESULT>1</RESULT>. No explanation or anything else."
    )
    with open('prompts.txt', 'a') as prompt_file:
        prompt_file.write(f"Prompt Final: {input_message}\n")
    
    chat_completion_final = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        temperature=0.1,
        top_p=0.9
    )
    
    result = chat_completion_final.choices[0].message.content
    
    with open('outputs.txt', 'a') as prompt_file:
        prompt_file.write(f"\n{step}: {label} -> Output:\n{result}\n")
    
    start_marker = "<RESULT>"
    
    start_index = result.find(start_marker) + len(start_marker)
    
    if start_index != -1:
        predicted_item = result[start_index:].strip()
    else:
        raise ValueError("Failed to extract items")
    
    return predicted_item


def predict_item(meta_description, label, items, max_tokens, amount, step):
    possible_items = []
    avg_token_length = 4
    max_tokens -= 200 # To have a decent margin in case somethings are added to the prompt after the fact (completion)
    item_length = len(items["Intitulé"])

    if len(items) * item_length / avg_token_length > max_tokens:
        num_batches = len(items) * item_length // (avg_token_length * max_tokens) + 1
        batch_size = len(items) // num_batches

        for i in range(num_batches):
            print("ACTUALLY U ARE IN " + f"{i}/{num_batches}")
            start = i * batch_size
            end = (i + 1) * batch_size if i != num_batches - 1 else len(items) # Otherwise it skips the last 2-3 items
            batch_items = items.iloc[start: end]
            context = prepare_context(batch_items)
            predicted_items = predict_items(context, meta_description, amount, label ,step)
            for predicted_item in predicted_items:
                possible_items.append(predicted_item)
    else:
        possible_items = items
    print("\nACTUALLY U ARE IN FINAL ROUND:")
    # Generate a final prediction
    filtered_items = items[items['Item'].isin(possible_items)].sample(frac=1).reset_index(drop=True) # Randomizes the order so that LLM does not pick up on unwanted patterns
    context = prepare_context(filtered_items)
    predicted_item = predict_final_item(context, meta_description, label, step)
    return predicted_item

def match_obj_toItem(questions, items, max_tokens, amount):
    """Matches learning objectives to items"""
    items_dict = {row['Item']: row['Intitulé'] for _, row in items.iterrows()}
    results = []
    step = 1

    with open('results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['learning_objective', 'description', 'predictedItem', 'itemTitle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()

        for _, row in questions.iterrows():
            try:
                meta_description = row['description']
                label = row['label']
                predicted_item = predict_item(meta_description, label, items, max_tokens, amount, step)

                predicted_item_num = re.findall(r'\d+', predicted_item)[0]  # Remove the extra characters from LLM's output
                itemTitle = items_dict.get(predicted_item_num, "Unknown Item")

                writer.writerow({
                    'learning_objective': label,
                    'description': meta_description,
                    'predictedItem': predicted_item_num,
                    'itemTitle': itemTitle
                })
                # Flush the buffer (otherwise the file doesn't update while the script is running)
                csvfile.flush()

                print(f"\n {step}: {label} -> Item Number: {predicted_item_num} ({itemTitle})\n")
                results.append(predicted_item_num)
            except Exception as e:
                writer.writerow({
                    'learning_objective': label,
                    'description': meta_description,
                    'predictedItem': '0',
                    'itemTitle': 'Item Not Found'
                })
                # Flush the buffer (otherwise the file doesn't update while the script is running)
                csvfile.flush()
                
                print(f"Failed to predict a valid Item for {label}. Using default. Error: {e}")
                results.append('0')
            step += 1
    return results


max_tokens = 8000
amountItems = 3 # Nbre of items to choose out of each batch
# sampled_meta_df = meta_df.sample(n=10, random_state=42)
match_obj_toItem(meta_df, items_df, max_tokens, amountItems)
