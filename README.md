Data and code for the paper "Yes, no, maybe? Revisiting language models' response stability under paraphrasing for the assessment of political leaning"

## Installation
- Prerequisites: Python >= 3.8, PyTorch
- `pip install -r requirements.txt`

## Data

### Political Compass questions
The directory `data/test_statements` contains the following files:
- `compass_eng.jsonl`: The original 62 questions from the [Political Compass Test](https://www.politicalcompass.org/test).
- `paraphrases_gpt3.5-turbo_temp1.json`: 500 paraphrases generated by OpenAI's GPT-3.5-turbo for the 62 original questions.
- `paraphrases_gpt3.5-turbo_temp1.json`: 500 paraphrases generated by Anthropic's Claude-3-5-Sonnet for the 62 original questions.

### Model responses
- The directory `data/responses`contains all model responses as in json-format.
- The directory `data/final_df`contains the all model responses in one csv files per LM type with the additional annotations. The files are too large for the repository, but they can be created using `get_text_features.py`(see below).

## Scripts

### Paraphrase generation
- `extract_paraphrases_openai.py`: Generate paraphrases for original statements in `data/test_statements/compass_eng.jsonl` using the openAI API (requires API key).
- `extract_paraphrases_anthropic.py`: Generate paraphrases for original statements `data/test_statements/compass_eng.jsonl` using the anthropic API (requires API key).

### Generate model responses
`python src/generate_responses.py --lm <model_name> --statements <set_of_statements>`
- `--lm`: `bert-base / bert-large / distilbert / distilroberta / electra-small / roberta-base / roberta-large / falcon-7b / falcon-7b-instruct / llama-7b / llama-7b-chat / llama-13b / llama-13b-chat / phi2 / tinyllama / gpt-3.5 / gpt-4 / gpt-4o`
- `--statements`: `paraphrases_gpt-35 / paraphrases_claude-sonnet / test`

### Extract additional features
- `get_text_features.py`: Extract additional features (word frequencies, sentiment, etc.) and write results from all models into final data frame (`data/final_df`)

### Analysis
- `analyze.R`: All tables and figures can be reproduced using this R-script. 