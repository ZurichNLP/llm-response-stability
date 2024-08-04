import concurrent.futures
import logging
import json
import numpy as np

from openai import OpenAI
from pathlib import Path
from random import sample
from tqdm import tqdm


client = OpenAI()

# Set up logging configuration
logging.basicConfig(filename='api_warnings.log', level=logging.WARNING)

# set seed
np.random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
statement_file = REPO_ROOT / "data/test_statements/compass_eng.jsonl"

# number of parallel API calls (each call generates 20 paraphrases, so 30 calls generate 600 paraphrases)
num_workers = 30

def run_completion(statement, j):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=1,
        seed = j,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to create paraphrases and output them separated by new lines."},
            {"role": "user", "content": f"Provide 20 paraphrases for the following statement: {statement}"}
        ]
    )
    paraphrases = completion.choices[0].message.content
    paraphrases = paraphrases.split("\n")
    # remove enumeration at the beginning of each paraphrase
    alternatives = []
    for paraphrase in paraphrases:
        paraphrase = paraphrase.split(". ")
        paraphrase = paraphrase[1:]
        paraphrase = " ".join(paraphrase)
        alternatives.append(paraphrase)
    return alternatives

def deduplicate_and_sample(alternative_statements, id):
    # deduplicate
    alternative_statements = list(set(alternative_statements))
    # remove empty strings
    alternative_statements = [alt for alt in alternative_statements if alt != ""]
    if len(alternative_statements) > 500:
        # randomly sample 500
        alternative_statements = sample(alternative_statements, 500)
    else:
        logging.warning(f"Less than 500 alternatives for statement {id}: {len(alternative_statements)}")

    return alternative_statements

if __name__ == "__main__":
    statements = json.loads(open(statement_file, "r").read())
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in tqdm(range(len(statements))):
            alternative_statements = []
            statement = statements[i]['statement']
            # rename statement key to "original_statement"
            statements[i]['original_statement'] = statements[i].pop('statement')
            # add id
            statements[i]['id'] = i
            statement_alternatives = []
            futures = [executor.submit(run_completion, statement, j) for j in range(num_workers)]
            for future in concurrent.futures.as_completed(futures):
                alternative_statements.extend(future.result())
            alternative_statements = deduplicate_and_sample(alternative_statements, i)
            statements[i]['alternative_statements'] = alternative_statements

    with open(f"data/test_statements/paraphrases_gpt3.5-turbo_temp1.json", "w") as f:
        json.dump(statements, f, indent=4)