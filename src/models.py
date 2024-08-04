import numpy as np
import os
import torch

from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer 
from transformers import RobertaTokenizer, DistilBertTokenizer, BertTokenizerFast

import constants as const
from helpers import initialize_dict

CACHE_DIR = os.environ.get('CACHE_DIR')
os.environ['OPENAI_API_KEY'] = const.OPENAI_KEY
client = OpenAI(timeout=150)
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_name, prompt_template):
    if model_name == "roberta-base":
        return RobertaBaseGenerator()
    elif model_name == "roberta-large":
        return RobertaLargeGenerator()
    elif model_name == "bert-base":
        return BERTBaseGenerator()
    elif model_name == "bert-large":
        return BERTLargeGenerator()
    elif model_name == "distilbert":
        return DistilBERTGenerator()
    elif model_name == "distilroberta":
        return DistilRobertaGenerator()
    elif model_name == "electra-small":
        return ElectraSmallGenerator()
    elif model_name == "gpt-3.5":
        return GPT35Generator(prompt_template=prompt_template)
    elif model_name == "gpt-4":
        return GPT4Generator(prompt_template=prompt_template)
    elif model_name == "gpt-4o":
        return GPT4oGenerator(prompt_template=prompt_template)
    elif model_name == "llama-7b":
        return Llama7Generator(prompt_template=prompt_template)
    elif model_name == "llama-7b-chat":
        return Llama7ChatGenerator(prompt_template=prompt_template)
    elif model_name == "llama-13b":
        return Llama13Generator(prompt_template=prompt_template)
    elif model_name == "llama-13b-chat":
        return Llama13ChatGenerator(prompt_template=prompt_template)
    elif model_name == "llama-70b":
        return Llama70Generator(prompt_template=prompt_template)
    elif model_name == "llama-70b-chat":
        return Llama70ChatGenerator(prompt_template=prompt_template)
    elif model_name == "llama-3-8b":
        return Llama3_8BGenerator(prompt_template=prompt_template)
    elif model_name == "llama-3-8b-instruct":
        return Llama3_8BInstructGenerator(prompt_template=prompt_template)
    elif model_name == "tinyllama":
        return TinyLlamaGenerator(prompt_template=prompt_template)
    elif model_name == "phi2":
        return Phi2Generator(prompt_template=prompt_template)
    elif model_name == "falcon-7b":
        return Falcon7BGenerator(prompt_template=prompt_template)
    elif model_name == "falcon-7b-instruct":
        return Falcon7BInstructGenerator(prompt_template=prompt_template)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class MLMGenerator:
    """
    Base class for MLM generators
    """
    def __init__(self):
        self.name = self.get_name()
        self.mask_token = self.get_mask_token()
        self.fill_mask_pipeline = self.get_pipeline()
        self.type = "mlm"
        
    def get_pipeline(self):
        return NotImplementedError("Subclasses must implement this method")
    
    def get_tokenizer(self):
        return NotImplementedError("Subclasses must implement this method")
    
    def get_name(self):
        return NotImplementedError("Subclasses must implement this method")
    
    def get_responses(self):
        return NotImplementedError("Subclasses must implement this method")
    
    # function that takes as input a list of dictionaries, searches for the key "token_str" and returns the value of the key "score" if they key is in "agree" or "disagree"
    def get_score(self, result: List) -> Dict[str, float]:
        scores = initialize_dict()
        for item in result:
            if item["token_str"].strip() in const.agree_list_mlm:
                scores['agree'] += item["score"]
            elif item["token_str"].strip() in const.disagree_list_mlm:
                scores['disagree'] += item["score"]
        return scores

    def get_prob_of_statement(self, statement):
        """
        Not implemented
        """
        return 0
    
    def get_responses(self, statement_file: List[Dict[str, str]]) -> List[Dict[str, str]]:
        unmasker = self.fill_mask_pipeline
        prompt = const.mlm_prompt
        prompt = prompt.replace("<mask>", self.mask_token)
        # prompt = "Please respond to the following statement: <statement> I <mask> with this statement." with either "agree" or "disagree", etc

        for i in tqdm(range(len(statement_file))):
        # for i in range(len(statement_file)):
            original_statement = statement_file[i]["original_statement"]
            alternative_statements = statement_file[i]["alternative_statements"]
            # convert the original statement to a value for a new key "statement" for statement_file[i]["statement_original"]
            statement_file[i]["original_statement"] = {}
            statement_file[i]["original_statement"]["statement"] = original_statement 
            statement_file[i]["original_statement"]["response"] = {}
            
            replaced = prompt.replace("<statement>", original_statement)
            result = unmasker(replaced)
            # get score
            scores = self.get_score(result)
            log_prob = self.get_prob_of_statement(original_statement)
            # add to statement_file
            statement_file[i]["original_statement"]["response"] = scores
            statement_file[i]["original_statement"]["statement_log_prob"] = log_prob

            for j in range(len(alternative_statements)):
                alternative_statement = alternative_statements[j]
                statement_file[i]["alternative_statements"][j] = {}
                statement_file[i]["alternative_statements"][j]["statement"] = alternative_statement
                statement_file[i]["alternative_statements"][j]["response"] = {}

                replaced = prompt.replace("<statement>", alternative_statement)
                result = unmasker(replaced)
                # get score
                scores = self.get_score(result)
                log_prob = self.get_prob_of_statement(alternative_statement)
                # add to statement_file
                statement_file[i]["alternative_statements"][j]["response"] = scores
                statement_file[i]["alternative_statements"][j]["statement_log_prob"] = log_prob

        return statement_file
  

class RobertaBaseGenerator(MLMGenerator):
    def get_name(self) -> str:
        return "roberta-base"
    
    def get_mask_token(self) -> str:
        return RobertaTokenizer.from_pretrained(self.name).mask_token
    
    def get_pipeline(self) -> pipeline:
        return pipeline("fill-mask", model = self.name, device = device, top_k = 20)
    

class RobertaLargeGenerator(MLMGenerator):
    def get_name(self) -> str:
        return "roberta-large"
    
    def get_mask_token(self) -> str:
        return RobertaTokenizer.from_pretrained(self.name).mask_token
    
    def get_pipeline(self) -> pipeline:
        return pipeline("fill-mask", model = self.name, device = device, top_k = 20)
    

class BERTBaseGenerator(MLMGenerator):
    def get_name(self) -> str:
        return "bert-base-uncased"
    
    def get_mask_token(self) -> str:
        return BertTokenizerFast.from_pretrained(self.name).mask_token
    
    def get_pipeline(self) -> pipeline:
        return pipeline("fill-mask", model = self.name, device = device, top_k = 20)
    

class BERTLargeGenerator(MLMGenerator):
    def get_name(self) -> str:
        return "bert-large-uncased"
    
    def get_mask_token(self) -> str:
        return BertTokenizerFast.from_pretrained(self.name).mask_token
    
    def get_pipeline(self) -> pipeline:
        return pipeline("fill-mask", model = self.name, device = device, top_k = 20)
    

class DistilBERTGenerator(MLMGenerator):
    def get_name(self) -> str:
        return "distilbert-base-uncased"
    
    def get_mask_token(self) -> str:
        return DistilBertTokenizer.from_pretrained(self.name).mask_token
    
    def get_pipeline(self) -> pipeline:
        return pipeline("fill-mask", model = self.name, device = device, top_k = 20)


class DistilRobertaGenerator(MLMGenerator):
    def get_name(self) -> str:
        return "distilroberta-base"
    
    def get_mask_token(self) -> str:
        return AutoTokenizer.from_pretrained(self.name).mask_token
    
    def get_pipeline(self) -> pipeline:
        return pipeline("fill-mask", model = self.name, device = device, top_k = 20)


class ElectraSmallGenerator(MLMGenerator):
    def get_name(self) -> str:
        return "google/electra-small-generator"
    
    def get_mask_token(self) -> str:
        return AutoTokenizer.from_pretrained(self.name).mask_token
    
    def get_pipeline(self) -> pipeline:
        return pipeline("fill-mask", model = self.name, device = device, top_k = 20)


class OpenAIGenerator:
    """
    Base class for Openai LMs
    """
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
        self.type = "glm"

    def get_responses(self, statement_file: List[Dict[str, str]]) -> List[Dict[str, str]]:
        prompt = self.prompt_template

        for i in tqdm(range(len(statement_file))):
            original_statement = statement_file[i]["original_statement"]
            alternative_statements = statement_file[i]["alternative_statements"]
            # convert the original statement to a value for a new key "statement" for statement_file[i]["statement_original"]
            statement_file[i]["original_statement"] = {}
            statement_file[i]["original_statement"]["statement"] = original_statement 
            statement_file[i]["original_statement"]["response"] = {} 

            # prob_statement = self.get_prob_of_statement(original_statement)
            scores = self.get_probabilities(prompt, original_statement)
            # add to statement_file
            statement_file[i]["original_statement"]["response"] = scores
            statement_file[i]["original_statement"]["statement_log_prob"] = 0  

            for j in range(len(alternative_statements)):
                alternative_statement = alternative_statements[j]
                statement_file[i]["alternative_statements"][j] = {}
                statement_file[i]["alternative_statements"][j]["statement"] = alternative_statement
                statement_file[i]["alternative_statements"][j]["response"] = {}

                # prob_statement = self.get_prob_of_statement(alternative_statement)
                scores = self.get_probabilities(prompt, alternative_statement)
                statement_file[i]["alternative_statements"][j]["response"] = scores
                statement_file[i]["alternative_statements"][j]["statement_log_prob"] = 0

        return statement_file

    def get_probabilities(self, prompt, statement):
        prompt_statement = prompt.replace("<statement>", statement)
        prompt_statement = prompt_statement.replace("<answer>", "")
        completion = client.chat.completions.create(
            model = self.name,
            # temperature = 1.0,
            logprobs = True,
            top_logprobs=20,
            messages=[
                {"role": "system", "content": f"{const.system_prompt}"},
                {"role": "user", "content": f"{prompt_statement}"}
            ]
        )
        scores = self.extract_scores(completion)
        return scores

    def extract_scores(self, completion):
        scores = initialize_dict()
        for item in completion.choices[0].logprobs.content[0].top_logprobs:
            if item.token.strip().lower() in const.agree_list_glm:
                scores['agree'] += np.exp(item.logprob)
            elif item.token.strip().lower() in const.disagree_list_glm:
                scores['disagree'] += np.exp(item.logprob)
        scores['agree'] = scores['agree']
        scores['disagree'] = scores['disagree']
        return scores


class GPT35Generator(OpenAIGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "gpt-3.5-turbo"


class GPT4Generator(OpenAIGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "gpt-4-turbo"


class GPT4oGenerator(OpenAIGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "gpt-4o"


class L2RGenerator:
    """
    Base class for L2R generators
    """
    def __init__(self, prompt_template):
        self.model = self.load_model()
        self.tokenizer = self.get_tokenizer()
        self.prompt_template = prompt_template
        self.STRIDE = 200
        self.type = "glm"

    def load_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_tokenizer(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def score(self, sentence, answer, return_all_probs=False, BOS=False):
        if return_all_probs:
            test = self.tokenizer('Test').input_ids
            if not test[0] == self.tokenizer.bos_token_id:
                BOS=True

        else:    
            # get number of subtokens in tokenized answer
            answer_subtokens = self.tokenizer(answer).input_ids
            # if first entry is the bos token, remove it, otherwise set BOS True
            if answer_subtokens[0] == self.tokenizer.bos_token_id:
                answer_subtokens = answer_subtokens[1:]
            else:
                BOS = True
            n_answer_subtokens = len(answer_subtokens)
            # For now, we only want to score sentences that have a single subtoken as the answer
            if n_answer_subtokens > 1:
                return None    
        
        with torch.no_grad():
            all_probs = torch.tensor([], device=self.model.device)
            offset_mapping = []
            start_ind = 0
            while True:
                encodings = self.tokenizer(
                    sentence[start_ind:],
                    max_length=1022,
                    truncation=True,
                    return_offsets_mapping=True,
                )
                if BOS:
                    tensor_input = torch.tensor(
                        [
                            [self.tokenizer.bos_token_id]
                            + encodings["input_ids"]
                            + [self.tokenizer.eos_token_id]
                        ],
                        device=self.model.device,
                    )
                else:
                    tensor_input = torch.tensor(
                        [encodings["input_ids"] ], # [self.tokenizer.eos_token_id]
                        device=self.model.device,
                    )
                output = self.model(tensor_input, labels=tensor_input)
                # shift logits because at each step k, the logits represent the probabilities of the tokens at step k+1
                # we remove the last token because it's the EOS token (-> to get the answer token with the highest probability, need to look at SECOND to last token in shifted logits)
                shift_logits = output["logits"][..., :-1, :].contiguous()
                shift_labels = tensor_input[..., 1:].contiguous()

                # get probabilities: softmax over shift logits 
                all_probabilities = torch.nn.functional.softmax(shift_logits, dim=-1)
                shift_labels_np = shift_labels.cpu().numpy()
                # Use advanced indexing to select the probabilities [batch (default 0), subword_idx, id_in_vocab]
                # shift_labels_np.shape[1] is just the number of subword tokens in the sentence
                subtoken_probabilities = all_probabilities[0, range(shift_labels_np.shape[1]), shift_labels_np[0]]

                if not return_all_probs:
                    # get probabilities that are part of the answer only
                    answer_subtoken_probabilities = subtoken_probabilities[(-n_answer_subtokens-1):-1]
                    answer_probabilities = torch.prod(answer_subtoken_probabilities)
                
                offset = 0 if start_ind == 0 else self.STRIDE - 1
                all_probs = torch.cat([all_probs, subtoken_probabilities[offset:-1]])
                offset_mapping.extend(
                    [
                        (i + start_ind, j + start_ind)
                        for i, j in encodings["offset_mapping"][offset:]
                    ]
                )
                if encodings["offset_mapping"][-1][1] + start_ind == len(sentence):
                    break
                start_ind += encodings["offset_mapping"][-self.STRIDE][1]
            if return_all_probs:
                return np.asarray(all_probs.cpu())
            else:
                return answer_probabilities.cpu().item()
        
    def get_responses(self, statement_file: List[Dict[str, str]]) -> List[Dict[str, str]]:
        prompt = self.prompt_template

        for i in tqdm(range(len(statement_file))):
            original_statement = statement_file[i]["original_statement"]
            alternative_statements = statement_file[i]["alternative_statements"]
            # convert the original statement to a value for a new key "statement" for statement_file[i]["statement_original"]
            statement_file[i]["original_statement"] = {}
            statement_file[i]["original_statement"]["statement"] = original_statement 
            statement_file[i]["original_statement"]["response"] = {} 
            # statement_file[i]["original_statement"]["statement_log_prob"] = 0 

            prob_statement = self.get_prob_of_statement(original_statement)
            scores = self.extract_scores(prompt, original_statement)
            # add to statement_file
            statement_file[i]["original_statement"]["response"] = scores
            statement_file[i]["original_statement"]["statement_log_prob"] = float(prob_statement)

            for j in range(len(alternative_statements)):
                alternative_statement = alternative_statements[j]
                statement_file[i]["alternative_statements"][j] = {}
                statement_file[i]["alternative_statements"][j]["statement"] = alternative_statement
                statement_file[i]["alternative_statements"][j]["response"] = {}
                # statement_file[i]["alternative_statements"][j]["statement_log_prob"] = 0

                prob_statement = self.get_prob_of_statement(alternative_statement)
                scores = self.extract_scores(prompt, alternative_statement)
                statement_file[i]["alternative_statements"][j]["response"] = scores
                statement_file[i]["alternative_statements"][j]["statement_log_prob"] = float(prob_statement)

        return statement_file
    
    def get_prob_of_statement(self, statement):
        scores_statement = self.score(statement, answer="", return_all_probs=True)
        # log transform and add up scores
        scores_statement = np.log(scores_statement)
        prob_statement = np.sum(scores_statement)
        return prob_statement
    
    def find_topk(self, statement, BOS=False):
        # first, remove the <answer> token from the prompt
        statement = statement.replace(self.prompt_replace, self.prompt_replacement)

        with torch.no_grad():
            start_ind = 0
            encodings = self.tokenizer(
                statement[start_ind:],
                max_length=1022,
                truncation=True,
                return_offsets_mapping=True,
            )
            # we don't need to add the EOS token here because we're only interested in the logits for the last token
            if BOS:
                tensor_input = torch.tensor(
                    [
                        [self.tokenizer.bos_token_id]
                        + encodings["input_ids"]
                    ],
                    device=self.model.device,
                )
            else:
                tensor_input = torch.tensor(
                    [encodings["input_ids"]],
                    device=self.model.device,
                )
            output = self.model(tensor_input, labels=tensor_input)
            
            logits = output.logits[0, -1]
            logprobs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_token_ids = torch.topk(logprobs, 20, dim=-1)

            # convert both to lists
            try:
                topk_probs = topk_probs.cpu().numpy().tolist()
            except TypeError:
                topk_probs.cpu().float().numpy().tolist()
            topk_tokens = [self.tokenizer.decode(token_id) for token_id in topk_token_ids.cpu().numpy().tolist()]

            return list(zip(topk_probs, topk_tokens))
    
    def extract_scores(self, prompt, statement):
            scores_agree = 0
            scores_disagree = 0
            replaced_statement = prompt.replace("<statement>", statement)
            highest_prob_answer = self.find_topk(replaced_statement)
            
            for prob, answer in highest_prob_answer:
                if answer.strip().lower() in const.agree_list_glm:
                    scores_agree += float(prob)
                elif answer.strip().lower() in const.disagree_list_glm:
                    scores_disagree += float(prob)

            scores = initialize_dict()
            scores['agree'] = scores_agree
            scores['disagree'] = scores_disagree
            return scores
    

class TinyLlamaGenerator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "tiny-llama-1b"
        self.prompt_replace = " <answer>"
        self.prompt_replacement = ""

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    

class Llama7Generator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llama-7b"
        self.prompt_replace = "<answer>"
        self.prompt_replacement = ""

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("/srv/scratch3/llm/Llama-2-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("/srv/scratch3/llm/Llama-2-7b-hf")
    

class Llama7ChatGenerator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llama-7b-chat"
        self.prompt_replace = "<answer>"
        self.prompt_replacement = "\n"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("/srv/scratch3/llm/Llama-2-7b-chat-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("/srv/scratch3/llm/Llama-2-7b-chat-hf")


class Llama13Generator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llama-13b"
        self.prompt_replace = "<answer>"
        self.prompt_replacement = ""

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("/srv/scratch3/llm/Llama-2-13b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("/srv/scratch3/llm/Llama-2-13b-hf")


class Llama13ChatGenerator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llama-13b-chat"
        self.prompt_replace = "<answer>"
        self.prompt_replacement = "\n"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("/srv/scratch3/llm/Llama-2-13b-chat-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("/srv/scratch3/llm/Llama-2-13b-chat-hf")


class Llama70Generator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llama-70b"
        self.prompt_replace = "<answer>"
        self.prompt_replacement = ""

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("/srv/scratch3/llm/Llama-2-70b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("/srv/scratch3/llm/Llama-2-70b-hf")


class Llama70ChatGenerator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llama-70b-chat"
        self.prompt_replace = "<answer>"
        self.prompt_replacement = "\n"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("/srv/scratch3/llm/Llama-2-70b-chat-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("/srv/scratch3/llm/Llama-2-70b-chat-hf")


class Llama3_8BGenerator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llama-3-8b"
        self.prompt_replace = " <answer>"
        self.prompt_replacement = ""

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
            token = const.HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token = const.HF_TOKEN)

class Llama3_8BInstructGenerator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "llama-3-8b-instruct"
        self.prompt_replace = "<answer>"
        self.prompt_replacement = "\n"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
            token = const.HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token = const.HF_TOKEN)


class Phi2Generator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "phi2"
        self.prompt_replace = " <answer>"
        self.prompt_replacement = "\n"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2", 
            device_map="auto",
            torch_dtype="auto", 
            trust_remote_code=True)
        # model.to(device)
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


class Falcon7BGenerator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "falcon-7b"
        self.prompt_replace = " <answer>"
        self.prompt_replacement = ""

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        tok = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        # had to fix this manually
        tok.bos_token_id = 11
        return tok


class Falcon7BInstructGenerator(L2RGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "falcon-7b-instruct"
        self.prompt_replace = " <answer>"
        self.prompt_replacement = "\n"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b-instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            )
        model.eval()
        return model
    
    def get_tokenizer(self):
        tok = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
        tok.bos_token_id = 11
        return tok