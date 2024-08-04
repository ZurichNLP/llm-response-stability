from typing import Dict
import csv
    
# function to initialize default dict with four keys: strongly agree, strongly disagree, disgree, strongly disagree with default value 0
def initialize_dict() -> Dict[str, float]:
    return {"agree": 0, "disagree": 0}

def save_to_csv(data, csv_file_path, model_name):
    # Define CSV headers
    csv_headers = ['model_name', 'statement_id', 'original_alternative', 'statement_str', 'agree', 'disagree', 'statement_log_prob']

    # Write data to CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write headers to CSV file
        csv_writer.writerow(csv_headers)

        # Write each record to CSV file
        for record in data:
            original_statement = record['original_statement']['statement']
            original_statement_log_prob = record['original_statement']['statement_log_prob']
            original_agree = record['original_statement']['response']['agree']
            original_disagree = record['original_statement']['response']['disagree']

            # Write original statement data to CSV file
            csv_writer.writerow([model_name, record['id'], 'original', original_statement, original_agree, original_disagree, original_statement_log_prob])

            for i, alternative_statement in enumerate(record['alternative_statements'], 1):
                alt_statement = alternative_statement['statement']
                alt_statement_log_prob = alternative_statement['statement_log_prob']
                alt_agree = alternative_statement['response']['agree']
                alt_disagree = alternative_statement['response']['disagree']

                # Write alternative statement data to CSV file
                csv_writer.writerow([model_name, record['id'], f'alternative_{i}', alt_statement, alt_agree, alt_disagree, alt_statement_log_prob])
        

    print(f'CSV file has been created: {csv_file_path}')