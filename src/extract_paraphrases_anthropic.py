import anthropic
import concurrent.futures
import json
import logging
import numpy as np
import time

from pathlib import Path
from random import sample
from tqdm import tqdm

client = anthropic.Anthropic(
    api_key=...
)

# Set up logging configuration
logging.basicConfig(filename='api_warnings.log', level=logging.WARNING)

# set seed
np.random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
statement_file = REPO_ROOT / "data/test_statements/compass_eng.jsonl"

num_workers = 7
model_str = "claude-3-5-sonnet-20240620"
outfile = f"data/test_statements/paraphrases_{model_str}_temp1.json"

def run_completion(statement, j):
    completion = client.messages.create(
        model=model_str,
        max_tokens=4000,
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Your task is to paraphrase a statement, creating 100 new paraphrases, and preserving the original meaning.\n\nHere is the original statement to paraphrase:\n\n{statement}\n\nNote that the paraphrases should be separated by newlines, but not enumerated or labeled in any other way."
                    }
                ]
            }
        ]
    )
    paraphrases = completion.content[0].text
    paraphrases = paraphrases.split("\n")
    # remove first two paraphrases: first one is prompt response, second one empty
    paraphrases = paraphrases[2:]
    return paraphrases

def deduplicate_and_sample(alternative_statements, id):
    # deduplicate
    alternative_statements = list(set(alternative_statements))
    # remove empty strings
    alternative_statements = [alt for alt in alternative_statements if alt != ""]
    if len(alternative_statements) > 500:
        # random sample 500
        alternative_statements = sample(alternative_statements, 500)
    else:
        logging.warning(f"Less than 500 alternatives for statement {id}: {len(alternative_statements)}")

    return alternative_statements

if __name__ == "__main__":
    try:
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
                time.sleep(30)
    except Exception as e:
        print(e)
        with open(f"data/test_statements/paraphrases_interrupted_{i}.json", "w") as f:
            json.dump(statements, f, indent=4)

    with open(outfile, "w") as f:
        json.dump(statements, f, indent=4)
