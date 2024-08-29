# LisaSheetPrediction
Code for associating UNESS database questions to their corresponding LISA sheets using Language Models (Large & Tiny).

## Content

### `learning_objective_item_mapping.py`
The script used to create the mapping needed for the first step during a sheet prediction.

### `llm_lisa_sheet_prediction.py`
This script contains the functions that are used to make Large Language Models predict the corresponding sheets.
- If you run this script instead of importing it, you'll automatically use the Llama-3.1-70b model on the UNESS database questions and store the results in a CSV file (`predictions.csv`), which you can eventually parse to modify the database.
- You can run it by using the following command:

  ```bash
  python3 llm_lisa_sheet_prediction.py <number_of_questions>

    <number_of_questions> represents how many questions the script should work on during that run (defaults to 500).

Note:

    The script picks up right from where it left off.
    Before running the script for the first time, you have to adapt the code so that it knows which database contains the questions you want it to work on (the initial code is adapted to a local database).

### `tlm_lisa_sheet_prediction.py`
This script contains the functions that are used to make Tiny Language Models predict the corresponding sheets.

### `prediction_validation.py`
This script tests the predictions using LLM-generated questions (by Cyprien) which can be found in the file: generated_questions.csv.

  - If you want to compare model(s) by creating a graph, you can run it by using the following command:

    ```bash

    python3 prediction_validation.py <number_of_questions> <model1>,<model2>,<model3>,...

        <number_of_questions>: represents how many questions the script should work on during that run.
        <model1>,<model2>,<model3>,...: represent the list (which can eventually contain only one) of models that you want to compare.

Note:

      If you pass only one model in the arguments, the script will only print out the performance rather than generating an unnecessary graph.
      The script also computes Rank A and Rank B accuracies, which can be found in the generated file: sheet_accuracies.txt.

### `results-withCorrections.csv`
This file contains the learning objective to Knowledge Item mapping, which was verified by a medical field expert (Olivier). The initial mapping before the verification is in the file: `results.csv`.
