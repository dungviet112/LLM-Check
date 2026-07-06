import json
import pickle
import datasets
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from common_utils import *

data_dir = Path("./data/")
model_dir = Path("../llm-hallucinations-factual-qa/.cache/models/")

def get_trivia_qa_data(n_samples=1000):
    trivia_qa = datasets.load_dataset('mandarjoshi/trivia_qa', data_dir='rc.nocontext', cache_dir=str(data_dir))
    full_dataset = []
    for obs in tqdm(trivia_qa['train']):
        aliases = []
        aliases.extend(obs['answer']['aliases'])
        aliases.extend(obs['answer']['normalized_aliases'])
        aliases.append(obs['answer']['value'])
        aliases.append(obs['answer']['normalized_value'])
        full_dataset.append((obs['question'], aliases))
    dataset = full_dataset[0: n_samples]
        
    return dataset, []

def get_scores_dict(model_name_or_path, data, mt_list, args):
    """
    Args:
        model_name_or_path (str): Path or identifier of the evaluator model
        data (list[dict]): Formatted data returned by get_trivia_qa_data()
        mt_list (list): List of metrics/methods to compute
        args (Namespace): Command-line arguments
    """
    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, 
                                                dtype=torch.bfloat16, 
                                                cache_dir=model_dir,
                                                attn_implementation="eager")
    tok_lens, scores, labels = [], [], []

    indiv_scores = {}
    for mt in mt_list:
        indiv_scores[mt] = defaultdict(def_dict_value)

    for i in tqdm(range(len(data))):
        question, answers = data[i]
        question_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output = model.generate(question_ids, max_new_tokens=100, temperature=0.1, do_sample=True)
        response_ids = output[0][question_ids.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        for alias in answers:
            if alias.lower() in response.lower():
                labels.append(1)
                break
        else:
            labels.append(0)

        full_text = question + " " + response
        full_text = full_text.replace("\n", " ").strip()

        tok_in_u = question_ids
        tok_in = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids
        tok_lens.append([tok_in_u.shape[1], tok_in.shape[1]])

        logit, hidden_act, attn = get_model_vals(model, tok_in.to(0))
        # Unpacking the values into lists on CPU
        logit = logit[0].cpu()
        hidden_act = [x[0].to(torch.float32).detach().cpu() for x in hidden_act]
        attn = [x[0].to(torch.float32).detach().cpu() for x in attn]
        tok_in = tok_in.cpu()

        tok_len = [tok_in_u.shape[1], tok_in.shape[1]]
        compute_scores(
            [logit],
            [hidden_act],
            [attn],
            scores,
            indiv_scores,
            mt_list,
            [tok_in],
            [tok_len],
            use_toklens=args.use_toklens,
        )
        print("Score:", scores[-1])

    return scores, indiv_scores, labels