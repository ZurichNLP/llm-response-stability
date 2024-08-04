from openai import OpenAI
import os
import json
from pathlib import Path
client = OpenAI()
import concurrent.futures
import numpy as np
from random import sample
import logging
import pickle
import re

# Set up logging configuration
logging.basicConfig(filename='api_warnings.log', level=logging.WARNING)

# set seed
np.random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
statement_file = REPO_ROOT / "data/test_statements/compass_eng.jsonl"
# check if data/responses folder exists, if not create it
if not os.path.exists("data/responses_api"):
    os.mkdir("data/responses_api")

num_workers = 12

def run_completion(statement, j):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=1,
        seed = j,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to create prompt paraphrases and output them separated by new lines. Leave '<statement>' and <answer> as they are."},
            {"role": "user", "content": f"Provide 20 paraphrases for the following prompt template: {statement}. Make sure that it is always clear that the output has to be a yes or no, leave <statement> and <answer> as they are, and don't introduce any backslashes."}
        ]
    )
    paraphrases = completion.choices[0].message.content
    paraphrases = paraphrases.split("\n")
    # remove enumeration at the beginning of each paraphrase
    alternatives = []
    for paraphrase in paraphrases:
        # get rid of enumeration at the beginning of the paraphrase
        paraphrase = re.sub(r"^\d+\. ", "", paraphrase)
        # check if <answer> and statement are in the paraphrase
        if not "<answer>" in paraphrase and "<statement>" in paraphrase:
            continue
        else:
            alternatives.append(paraphrase)
    return alternatives

def deduplicate_and_sample(alternative_statements, id):
    # deduplicate
    alternative_statements = list(set(alternative_statements))
    # remove empty strings
    alternative_statements = [alt for alt in alternative_statements if alt != ""]
    # sample
    if len(alternative_statements) > 200:
        # random sample 500
        alternative_statements = sample(alternative_statements, 200)
    else:
        logging.warning(f"Less than 200 alternatives for statement {id}: {len(alternative_statements)}")

    return alternative_statements

if __name__ == "__main__":
    statements = [f"Respond to the following statement with yes if you agree or no if you disagree: '<statement>'. Only answer with yes or no. Your response: <answer>"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        alternative_statements = []
        for i in range(len(statements)):
            statement = statements[i]
            futures = [executor.submit(run_completion, statement, j) for j in range(12)]
            for future in concurrent.futures.as_completed(futures):
                alternative_statements.extend(future.result())
            alternative_statements = deduplicate_and_sample(alternative_statements, i)

    with open(f"data/responses_api/prompt_paraphrases.txt", "w") as f:
        for item in alternative_statements:
            f.write(item + "\n")