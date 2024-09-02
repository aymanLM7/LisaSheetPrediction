"""Script to validate the LISA sheet prediction"""
import ast
import pandas as pd
import sys
import matplotlib.pyplot as plt
from groq import Groq
from llm_lisa_sheet_prediction import match_questions_to_sheets, summarize_text


questions_csv_path = 'generated_questions.csv'
client = Groq(api_key= "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") # TODO remove

questions_df = pd.read_csv(questions_csv_path, usecols=['question_id', 'question_text', 'answers', 'correction', 'lisa_sheet_id'])

questions_df.rename(columns={'question_text': 'questiontext'}, inplace=True)

questions_df['description'] = 'Oops this question has no description.'  # No description for generated questions


def get_correct_answer(question):
    """Get ONE of the correct answers of the question"""
    result = 'Oops couldnt find any answers'
    if isinstance(question['answers'], str):
        result = ''
        answers = question['answers'].split(';;')
        correction = ast.literal_eval(question['correction']) # correction is now a list of floats and each float is the grading of the corresponding answer
        
        for i, value in enumerate(correction):
            if value > 0:
                result += answers[i] + 'sep%'
    return result[:-4] # To remove the last 'sep%'


def generate_description(question):
    """Helper func that generates a description in case you want to add it in the prompt"""
    with open(f"lisa_sheets/{question['lisa_sheet_id'][1:7]}/{question['lisa_sheet_id']}.txt", 'r') as f:
        content = f.read()
        sheet = content.split('|')[5].strip()
        item = content.split('|')[2].strip().split('=', maxsplit=1)[-1]
    input_message = (
        f"Votre tâche est de fournir une description de la question écrite en FRANÇAIS pour aider quelqu'un à prédire l'item et la feuille LISA (en particulier l'item) de la question médicale donnée. "
        f"Répondez uniquement par : Description : <Description>.\n"
        f"Ne fournissez aucune explication ni texte supplémentaire.\n\n"
        f"N'incluez pas l'ID exact et/ou le numéro de la feuille LISA et/ou de l'item, ni le nom exact (vous pouvez les paraphraser et utiliser des mots-clés).\n"
        f"Question : {question['questiontext']}\n"
        f"Item LISA : {sheet}\n"
        f"Feuille LISA : {item}\n"
        f"Assurez-vous qu'il est facile de prédire de quel item et de quelle feuille LISA il s'agit !"
    )
    with open('description_ prompts.txt', 'a') as prompt_file:
        prompt_file.write('\n' + input_message + '\n')

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="llama3-8b-8192",
        temperature=0.1,
        top_p=0.9
        )

    result = chat_completion.choices[0].message.content.split('Description', maxsplit=1)[-1][2:]
    with open('description_ outputs.txt', 'a') as prompt_file:
        prompt_file.write('\n' + result + '\n')
   
    if result:
        print(f"Description : {result}")
        return result
    else:
        raise ValueError(f"API request failed.")

def add_description(questions):
    """Add a description about the LISA sheet of a generated question"""
    for indice, question in questions.iterrows():
        description = summarize_text(generate_description(question), max_length=200)
        questions.at[indice, 'description'] = description


def compute_accuracy(nb_questions, questions_df, max_tokens, chosenModel):
    """Computes the general accuracy and the accuracy for rank A and rank B questions"""
    sampled_questions_df = questions_df.sample(n=nb_questions, random_state=546)
    sampled_questions_df['correct_answer'] = sampled_questions_df.apply(get_correct_answer, axis=1)
    
    predictions_df = match_questions_to_sheets(sampled_questions_df, max_tokens, True, chosenModel)  # 8K tokens for Llama-3-8B
    print(sampled_questions_df)
    print(predictions_df)

    merged_df = pd.merge(sampled_questions_df, predictions_df, on='question_id')
    merged_df['correct_pred'] = merged_df['lisa_sheet_id'] == merged_df['predicted_sheet_id']

    precision = merged_df['correct_pred'].mean() * 100  # Percentages are easier to read

    rank_A_df = merged_df[merged_df['lisa_sheet_id'].str.endswith('A')]
    rank_A_precision = rank_A_df['correct_pred'].mean() * 100 if not rank_A_df.empty else 0

    # Calculate precision for rank B
    rank_B_df = merged_df[merged_df['lisa_sheet_id'].str.endswith('B')]
    rank_B_precision = rank_B_df['correct_pred'].mean() * 100 if not rank_B_df.empty else 0

    return precision, rank_A_precision, rank_B_precision


def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) < 3:
        print("Usage: python script.py <nb_questions> <model1>,<model2>,<model3>,...")
        sys.exit(1)

    # Parse the arguments
    nb_questions = int(sys.argv[1])
    models = sys.argv[2].split(",")

    # Dictionary to store the precision results
    precision_dict = {}

    for chosenModel in models:
        precision, rank_A_precision, rank_B_precision = compute_accuracy(nb_questions, questions_df, 8000, chosenModel)
        print(f'nb_questions: {nb_questions} -> Precision: {precision:.2f} - Model: {chosenModel}')
        
        # Store the precision in the dictionary
        precision_dict[chosenModel] = precision

        # Write the precision in a file
        with open('sheet_accuracies.txt', 'a') as file:
            file.write(f'nb_questions: {nb_questions} -> Precision: {precision:.2f} , Rank A Precision: {rank_A_precision:.2f} , Rank B Precision: {rank_B_precision:.2f} - Model: {chosenModel}\n-----\n')
    if len(models) == 1:
        return

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.bar(precision_dict.keys(), precision_dict.values(), color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Performance (%)')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 100) 
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
