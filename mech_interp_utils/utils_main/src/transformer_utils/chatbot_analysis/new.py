from datasets import load_dataset

def load_nq_subset(split='train', sample_size=200):
    dataset = load_dataset("natural_questions", split=split)
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    return dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def compute_exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)
def evaluate_models_on_nq(models, tokenizer, save_path, sample_size=200):
    nq_dataset = load_nq_subset(sample_size=sample_size)
    results = []

    for example in tqdm(nq_dataset):
        prompt = example["question"]["text"]
        context = example["document"]["text"]
        # Use first short answer if available
        if example["annotations"][0]["short_answers"]:
            gt_answer = example["annotations"][0]["short_answers"][0]["text"]
        else:
            continue  # skip if no answer available

        for model_name, model in models.items():
            res = _run_chatbot_analysis(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                context=context,
                prompt=prompt,
                max_new_tokens=30,
                temperature=0.7,
                repetition_penalty=1.0,
                sample=False,
                device='cuda',
                save_path=save_path,
                deterministic_backend=True
            )
            pred = res["Sample Response"]
            em = compute_exact_match(pred, gt_answer)
            f1 = compute_f1(pred, gt_answer)

            res["Ground Truth"] = gt_answer
            res["Exact Match"] = em
            res["F1 Score"] = f1
            res["Question"] = prompt

            results.append(res)

    return pd.DataFrame(results)
