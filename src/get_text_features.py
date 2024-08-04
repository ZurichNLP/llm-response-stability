import argparse
import concurrent.futures
import json
import os
import pandas as pd
import string

from Levenshtein import ratio
from pysentimiento import create_analyzer
from tqdm import tqdm
from wordfreq import word_frequency, zipf_frequency

from dependencies import DependencyParser
import pdb

parser = argparse.ArgumentParser(description="Get additional fext features.")
parser.add_argument(
    "--responses", 
    dest="responses", 
    help="Path to the responses json file.",
    choices=["gpt_paraphrases/glm", "gpt_paraphrases/mlm", "claude_paraphrases/glm", "claude_paraphrases/mlm"]
)
args = parser.parse_args()

response_dir = f"data/responses/{args.responses}"
response_dir_outname = args.responses.replace('/', '_')
outfile_name = f'data/final_df/all_models_{response_dir_outname}_parallel.csv'

analyzer = create_analyzer(task="sentiment", lang="en")
dep_parser = DependencyParser()
# check if data/final_df exists else create it
if not os.path.exists('data/final_df'):
    os.makedirs('data/final_df')

def get_sentiment(text):
    """
    Get the sentiment of a text.
    """
    return analyzer.predict(text)

def normalized_levenshtein_distance(s1, s2):
    """
    Calculate the normalized Levenshtein distance between two strings.
    """
    return ratio(s1, s2)

def average_zipf_freq(sentence):
    """
    Calculate the average zipf frequency of words in a sentence.
    """
    # remove all punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sum = 0
    for word in sentence.split():
        sum += zipf_frequency(word, 'de')
    return sum / len(sentence.split())

def average_word_freq(sentence):
    """
    Calculate the average lexical frequency of words in a sentence.
    """
    # remove all punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sum = 0
    for word in sentence.split():
        sum += word_frequency(word, 'de')
    return sum / len(sentence.split())


def average_word_length(sentence):
    """
    Calculate the average word length of a sentence.
    """
    # remove all punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sum(len(word) for word in sentence.split()) / len(sentence.split())

def sentence_length(sentence):
    """
    Calculate the length of a sentence.
    """
    return len(sentence.split())


def extract_information_from_json(json_file, model_name, prompt='default'):
    extracted_info = {
        'model': [],
        'prompt': [],
        'original_statement': [],
        'alternative_statement': [],
        'distance_to_original_statement': [],
        'average_lex_freq_original': [],
        'average_word_length_original': [],
        'sentence_length_original': [],
        'average_lex_freq_alternative': [],
        'average_word_length_alternative': [],
        'sentence_length_alternative': [],
        'n_rights_original': [],
        'n_lefts_original': [],
        'dep_distance_original': [],
        'n_rights_alternative': [],
        'n_lefts_alternative': [],
        'dep_distance_alternative': [],
        'sentiment_pos_original': [],
        'sentiment_neg_original': [],
        'sentiment_neu_original': [],
        'sentiment_pos_alternative': [],
        'sentiment_neg_alternative': [],
        'sentiment_neu_alternative': [],
        'agree_original' : [],
        'agree_alternative' : [],
        'disagree_original' : [],
        'disagree_alternative' : [],
        'log_prob_original' : [],
        'log_prob_alternative' : [],
        'original_alternative': [],
        'id': []
    }

    with open(json_file) as f:
        data = json.load(f)

    for item in tqdm(data):
        id = item['id']

        original_statement = item['original_statement']['statement']
        n_rights_original, n_lefts_original, dep_distance_original = dep_parser.parse_dependency(original_statement)
        sentiment_original = get_sentiment(original_statement)
        extracted_info['model'].append(model_name)
        extracted_info['prompt'].append(prompt)
        extracted_info['id'].append(id)
        extracted_info['original_statement'].append(original_statement)
        extracted_info['alternative_statement'].append('na')
        extracted_info['distance_to_original_statement'].append(0)
        extracted_info['average_lex_freq_original'].append(average_zipf_freq(original_statement))
        extracted_info['average_word_length_original'].append(average_word_length(original_statement))
        extracted_info['sentence_length_original'].append(sentence_length(original_statement))
        extracted_info['average_lex_freq_alternative'].append('na')
        extracted_info['average_word_length_alternative'].append('na')
        extracted_info['sentence_length_alternative'].append('na')
        extracted_info['n_rights_original'].append(n_rights_original)
        extracted_info['n_lefts_original'].append(n_lefts_original)
        extracted_info['dep_distance_original'].append(dep_distance_original)
        extracted_info['n_rights_alternative'].append('na')
        extracted_info['n_lefts_alternative'].append('na')
        extracted_info['dep_distance_alternative'].append('na')
        extracted_info['sentiment_pos_original'].append(sentiment_original.probas['POS'])
        extracted_info['sentiment_neg_original'].append(sentiment_original.probas['NEG'])
        extracted_info['sentiment_neu_original'].append(sentiment_original.probas['NEU'])
        extracted_info['sentiment_pos_alternative'].append('na')
        extracted_info['sentiment_neg_alternative'].append('na')
        extracted_info['sentiment_neu_alternative'].append('na')
        extracted_info['agree_original'].append(item['original_statement']['response']['agree'])
        extracted_info['agree_alternative'].append('na')
        extracted_info['disagree_original'].append(item['original_statement']['response']['disagree'])
        extracted_info['disagree_alternative'].append('na')
        extracted_info['log_prob_original'].append(item['original_statement']['statement_log_prob'])
        extracted_info['log_prob_alternative'].append('na')
        extracted_info['original_alternative'].append("original")

        for alt_statement in item['alternative_statements']:
            alternative_statement = alt_statement['statement']
            distance = normalized_levenshtein_distance(original_statement, alternative_statement)
            avg_lex_freq = average_zipf_freq(alternative_statement)
            n_rights_alternative, n_lefts_alternative, dep_distance_alternative = dep_parser.parse_dependency(alternative_statement)
            sentiment_alternative = get_sentiment(alternative_statement)
            extracted_info['model'].append(model_name)
            extracted_info['prompt'].append(prompt)
            extracted_info['id'].append(id)
            extracted_info['original_statement'].append(original_statement)
            extracted_info['alternative_statement'].append(alternative_statement)
            extracted_info['distance_to_original_statement'].append(distance)
            extracted_info['average_lex_freq_original'].append(average_zipf_freq(original_statement))
            extracted_info['average_word_length_original'].append(average_word_length(original_statement))
            extracted_info['sentence_length_original'].append(sentence_length(original_statement))
            extracted_info['average_lex_freq_alternative'].append(avg_lex_freq)
            extracted_info['average_word_length_alternative'].append(average_word_length(alternative_statement))
            extracted_info['sentence_length_alternative'].append(sentence_length(alternative_statement))
            extracted_info['n_rights_original'].append(n_rights_original)
            extracted_info['n_lefts_original'].append(n_lefts_original)
            extracted_info['dep_distance_original'].append(dep_distance_original)
            extracted_info['n_rights_alternative'].append(n_rights_alternative)
            extracted_info['n_lefts_alternative'].append(n_lefts_alternative)
            extracted_info['dep_distance_alternative'].append(dep_distance_alternative)
            extracted_info['sentiment_pos_original'].append(sentiment_original.probas['POS'])
            extracted_info['sentiment_neg_original'].append(sentiment_original.probas['NEG'])
            extracted_info['sentiment_neu_original'].append(sentiment_original.probas['NEU'])
            extracted_info['sentiment_pos_alternative'].append(sentiment_alternative.probas['POS'])
            extracted_info['sentiment_neg_alternative'].append(sentiment_alternative.probas['NEG'])
            extracted_info['sentiment_neu_alternative'].append(sentiment_alternative.probas['NEU'])
            extracted_info['agree_original'].append(item['original_statement']['response']['agree'])
            extracted_info['agree_alternative'].append(alt_statement['response']['agree'])
            extracted_info['disagree_original'].append(item['original_statement']['response']['disagree'])
            extracted_info['disagree_alternative'].append(alt_statement['response']['disagree'])
            extracted_info['log_prob_original'].append(item['original_statement']['statement_log_prob'])
            extracted_info['log_prob_alternative'].append(alt_statement['statement_log_prob'])
            extracted_info['original_alternative'].append(f"alternative")

    return extracted_info

def process_file(file_path):
    model_name = file_path.split('_')[2]
    extracted_data = extract_information_from_json(file_path, model_name)
    df = pd.DataFrame(extracted_data)
    return df

def main():
    file_paths = [f for f in os.listdir(response_dir) if f.endswith('.jsonl')]
    file_paths = [os.path.join(response_dir, file) for file in file_paths]
    results = []
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_file, file) for file in file_paths]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    df = pd.concat(results, ignore_index=True)
    df.to_csv(outfile_name, index=False)

if __name__ == "__main__":
    main()