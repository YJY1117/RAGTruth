import json
from collections import defaultdict
from pprint import pprint

# train dev test
with open("./test.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(i) for i in f]

# 初始化结果字典
results = {
    "by_model": {},
    "by_task_type": {}
}

# 遍历每条记录
for record in data:
    model = record["model"]
    task_type = record["task_type"]
    response_length = len(record["response"])
    reference_length = len(record["reference"])
    spans = record["labels"]
    spans_count = len(spans)
    spans_lengths = []

    for span in spans:
        if "start" in span and "end" in span:
            spans_lengths.append(span["end"] - span["start"])
        elif "text" in span:
            spans_lengths.append(len(span["text"]))

    # 模型统计
    if model not in results["by_model"]:
        results["by_model"][model] = {
            "response_count": 0,
            "spans_count": 0,
            "response_length_total": 0,
            "spans_length_total": 0,
            "reference_length_total": 0
        }

    model_stats = results["by_model"][model]
    model_stats["response_count"] += 1
    model_stats["spans_count"] += spans_count
    model_stats["response_length_total"] += response_length
    model_stats["spans_length_total"] += sum(spans_lengths)
    model_stats["reference_length_total"] += reference_length

    # 任务类型统计
    if task_type not in results["by_task_type"]:
        results["by_task_type"][task_type] = {
            "response_count": 0,
            "spans_count": 0,
            "response_length_total": 0,
            "spans_length_total": 0,
            "reference_length_total": 0
        }

    task_stats = results["by_task_type"][task_type]
    task_stats["response_count"] += 1
    task_stats["spans_count"] += spans_count
    task_stats["response_length_total"] += response_length
    task_stats["spans_length_total"] += sum(spans_lengths)
    task_stats["reference_length_total"] += reference_length

# 计算平均值
def compute_averages(stats):
    stats["response_avg_length"] = stats["response_length_total"] / stats["response_count"]
    stats["spans_avg_length"] = stats["spans_length_total"] / stats["spans_count"] if stats["spans_count"] > 0 else 0
    stats["reference_avg_length"] = stats["reference_length_total"] / stats["response_count"]

for model, model_stats in results["by_model"].items():
    compute_averages(model_stats)

for task_type, task_stats in results["by_task_type"].items():
    compute_averages(task_stats)

# 输出结果
pprint(results)

