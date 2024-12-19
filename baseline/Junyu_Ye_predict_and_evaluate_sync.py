from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import random
import traceback
import pandas as pd
import torch
from sklearn.metrics import recall_score, precision_score, f1_score
from dataset import process_dialog_to_single_turn

# from utils import get_short_ctx, get_short_ctx_embedding
parser = ArgumentParser()
parser.add_argument('--raw_dataset', default="./test.jsonl")
parser.add_argument('--output_file', default="./prediction.jsonl")
parser.add_argument('--model_name', default='baseline')
parser.add_argument('--tokenizer', default="meta-llama/Meta-Llama-3-8B")
parser.add_argument('--meta', action='store_true')
parser.add_argument('--fold', type=int, default=-1)
args = parser.parse_args()

embedder = None
B_INST, E_INST = "[INST]", "[/INST]"

# Load model and tokenizer synchronously
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
model = AutoModelForCausalLM.from_pretrained(args.tokenizer).cuda()
model.eval()

# Ensure pad_token_id is set to eos_token_id to avoid warnings
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(data):
    input_prompt = process_dialog_to_single_turn(data, tokenizer, return_prompt=True, train=False)
    input_prompt = f"{B_INST} {input_prompt.strip()} {E_INST}"

    # Tokenize input
    inputs = tokenizer(input_prompt, padding=True, return_tensors="pt")

    ret = dict(data)
    # Generate response
    try:
        outputs = model.generate(
            inputs['input_ids'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40
        )
        generated_text = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        # print("=============", type(outputs), generated_text, "=========");raise Exception
        answer = json.loads(generated_text)
    except Exception as e:
        print(f"Error during generation: {e.__class__} {e}")
        # traceback.print_exc()
        print(f"Input: {input_prompt}, {len(input_prompt)}=={len(generated_text)}")
        answer = {'non_json_text': generated_text}
        # print("===========\n", generated_text, "================");exit(-123)
    torch.cuda.empty_cache()
    ret['pred'] = answer
    return ret

def main(args):
    results = []
    bad_sample = 0

    with open(args.raw_dataset, 'r') as f:
        lines = f.readlines()

    pbar = tqdm(lines, desc="Processing")
    for line in pbar:
        data = json.loads(line)

        if args.fold >= 0 and data['fold'] != args.fold:
            continue

        result = generate_response(data)
        results.append(result)

    # Calculate metrics
    df = pd.DataFrame.from_records(results)
    df['is_halu'] = df['labels'].apply(lambda x: len(x)>0)
    df['pred_halu'] = df['pred'].apply(lambda x: len(x.get('hallucination list', []))>0)

    print(f"Case recall/precision/f1: {recall_score(df['is_halu'], df['pred_halu']):.3f}, {precision_score(df['is_halu'], df['pred_halu']):.3f}, {f1_score(df['is_halu'], df['pred_halu']):.3f}")
    for task in ['QA', 'Summary', 'Data2txt']:
        temp = df[df['task_type'] == task]
        print(f"{task}-Case recall/precision/f1: {recall_score(temp['is_halu'], temp['pred_halu']):.3f}, {precision_score(temp['is_halu'], temp['pred_halu']):.3f}, {f1_score(temp['is_halu'], temp['pred_halu']):.3f}")

    # Write results
    with open(args.output_file, 'w') as f:
        for result in results:
            if isinstance(result, dict):
                f.write(json.dumps(result) + "\n")
            else:
                bad_sample += 1

    print(f"Bad samples: {bad_sample}")

if __name__ == '__main__':
    main(args)
