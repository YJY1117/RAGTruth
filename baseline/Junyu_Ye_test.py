from transformers import AutoConfig, AutoModel
import torch
from transformers import pipeline

model_path = r"/Users/junyuye/Documents/learningspace/responsilble AI/project/RagTruth/baseline/exp/checkpoint-878/"

pipe = pipeline(
    "text-generation", 
    model=model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

print(pipe("The key to life is"))

# python Junyu_Ye_predict_and_evaluate_sync.py --model_name '/Users/junyuye/Documents/learningspace/responsilble AI/project/RagTruth/baseline/exp/checkpoint-878/' --tokenizer '/Users/junyuye/Documents/learningspace/responsilble AI/project/RagTruth/baseline/exp/checkpoint-878/' >> log.log
