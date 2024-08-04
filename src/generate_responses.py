import argparse
import json
import os

import models
import constants as const

parser = argparse.ArgumentParser(description="Run political biases experiment")
parser.add_argument("--lm", dest="lm")
parser.add_argument("--statements", dest="statements", default = "paraphrases_gpt-35")
parser.add_argument("--prompt", dest="prompt_template", default="default")
parser.add_argument("--outdir", dest="outdir", default="data/responses/gpt_paraphrases")
parser.add_argument("--no-write", dest="write", action="store_false")
parser.set_defaults(write=True)
args = parser.parse_args()

if args.statements in const.statement_files.keys():
    statement_file = const.statement_files[args.statements]
else:
    raise ValueError(f"Invalid statements argument: {args.statements}")

if args.prompt_template in const.prompt_templates.keys():
    prompt_template = const.prompt_templates[args.prompt_template]
else:
    raise ValueError(f"Invalid prompt_templates argument: {args.prompt_template}")

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

if __name__ == "__main__":
    statements = json.loads(open(statement_file, "r").read())
    current_model = models.get_model(args.lm, prompt_template)
    responses = current_model.get_responses(statements)
    
    if args.write:
        filename = f'responses_{current_model.name.replace("/", "-")}_{args.statements}_{args.prompt_template}'
        with open(f"{args.outdir}/{current_model.type}/{filename}.jsonl", "w") as f:
            json.dump(responses, f, indent = 4)
    else:
        print(responses)