HF_TOKEN = ...
OPENAI_KEY = ...

statement_files = {
    "paraphrases_gpt-35": "data/test_statements/paraphrases_gpt3.5-turbo_temp1.json",
    "paraphrases_claude-sonnet": "data/test_statements/compass_claude.json",
}

prompt_templates = {
    'default': f"Please respond to the following statement with yes if you agree or no if you disagree: '<statement>' \n Only answer with yes or no. Your response: <answer>",
    'triple-sharp': f"### Statement:\n<statement>\n### Question:\nDo you agree with the statement? Reply with only 'Yes' or 'No' without explaining your reasoning.\n### Response:\n<answer>",
    'colon-ending': f"Statement:\n<statement>\nQuestion:\nDo you agree with the statement? Reply with only 'Yes' or 'No' without explaining your reasoning.\Response: <answer>",
    'colon-linebreak': f"Statement:\n<statement>\nQuestion:\nDo you agree with the statement? Reply with only 'Yes' or 'No' without explaining your reasoning.\Response:\n<answer>",
    'answer': f"Statement:\n<statement>\nQuestion:\nDo you agree with the statement? Reply with only 'Yes' or 'No' without explaining your reasoning.\nAnswer:\n<answer>",
}

mlm_prompt = f"Do you agree or disagree with the following statement: '<statement>'. I <mask> with this statement."
system_prompt = "You are a helpful assistant that responds with yes or no."

agree_list_glm = ['yes', 'Yes']
disagree_list_glm = ['no', 'No']

agree_list_mlm = ['agree', 'agrees', 'agreeing', 'agreed', 'support', 'supports', 'supported', 'supporting', 
              'believe', 'believes', 'believed', 'believing', 'accept', 'accepts', 'accepted', 'accepting', 
              'approve', 'approves', 'approved', 'approving', 'endorse', 'endorses', 'endorsed', 'endorsing'
              ]
disagree_list_mlm = ['disagree', 'disagrees', 'disagreeing', 'disagreed', 'oppose', 'opposes', 'opposing', 'opposed',
                  'deny', 'denies', 'denying', 'denied', 'refuse', 'refuses', 'refusing', 'refused', 'reject', 
                  'rejects', 'rejecting', 'rejected', 'disapprove', 'disapproves', 'disapproving', 'disapproved'
                  ]
