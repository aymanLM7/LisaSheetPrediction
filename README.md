# LisaSheetPrediction
Code for associating Uness database questions to their corresponding LISA sheets using Language Models (Large & Tiny).


## Content

### `learning_objective_item_mapping.py`
The script used in order to make the mapping needed for the first step during a sheet prediction.

### `llm_lisa_sheet_prediction.py`
Script which contains the functions that are used to make Large Language Models predict the corresponding sheets.
- If you run this script instead of importing it, you'll automatically use the Llama-3.1-70b model on the UNESS database questions and store it in a csv file which you can eventually parse to modify the database

### `tlm_lisa_sheet_prediction.py`
Script which contains the functions that are used to make Tiny Language Models predict the corresponding sheets.

### `prediction_validation.py`
Script which tests the predictions using LLM generated questions (by Cyprien) which can be found in the file : 'generated_questions.csv'

### `results-withCorrections.csv`
Contains the learning_objective to Knowledge Item mapping which was verified by a medical field expert (Olivier). The initial mapping before the verification is in the file: 'results.csv'
