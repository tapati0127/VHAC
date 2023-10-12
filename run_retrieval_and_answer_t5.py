import csv
from typing import List

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from params.algo_params import C4IBLINKInferenceConfig
from src.model.blink_predictor import BLINKPredictor


def get_blink_format(question):
    return [{
        "context_left": "",
        "context_right": "",
        "mention": question,
        "label_title": "",
        "label": "",
        "label_id": -1
    }]


data_path = "data/data_private_question.csv"
with open(data_path, "r") as f:
    questions = list(csv.reader(f))
    questions = [(item[0], item[1]) for item in questions[1:]]

config = C4IBLINKInferenceConfig(top_candidates=40,
                                 bert_base_dir='xlm-roberta-base',
                                 biencoder_dir=
                                 'static/biencoder',
                                 crossencoder_dir=
                                 'static/crossencoder/cross',
                                 entity_catalogue_dir=
                                 'data',
                                 candidate_encode_dir=
                                 "static/")

predictor = BLINKPredictor(config)
model = AutoModelForSeq2SeqLM.from_pretrained("trientp/vit5_base_qa").to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("trientp/vit5_base_qa")


def ask(question, context):
    try:
        min_length = int(0.4 * len(context.split(" ")))
        max_length = int(1.5 * len(context.split(" ")))
        input_text = f"{question} </s> {context} </s>"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024)
        outputs = model.generate(
            input_ids=inputs['input_ids'].to('cuda'),
            max_length=max_length,
            min_length=min_length,
            attention_mask=inputs['attention_mask'].to('cuda'),
            no_repeat_ngram_size=3, num_beams=3
        )
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return text
    except Exception as ex:
        print(f"ERROR: {ex}")
        return question


def get_title_heading_content(text: str):
    ctx_list = text.split("\n", maxsplit=2)
    return ctx_list[0], ctx_list[1], ctx_list[2]


def get_context(contexts: List[str]):
    text = contexts[0]
    if len(contexts) == 1:
        return text
    title, heading, content = get_title_heading_content(contexts[0])
    for context in contexts[1:]:
        _title, _heading, _content = get_title_heading_content(context)
        if title == _title:
            if _heading == heading:
                text += "\n" + _content
            else:
                text += "\n" + _heading + "\n" + _content
        else:
            return text
    return text


results = [["quest_id", "answer_predict"]]
full_results = [["quest_id", "question", "context", "answer_predict"]]
for quest_id, question in tqdm(questions):
    contexts = predictor.get_contexts(get_blink_format(question), top_k=3)
    context = get_context(contexts)
    ans = ask(question, context)
    print("*" * 100)
    print(quest_id)
    print(question)
    print("*" * 30)
    print(context)
    print("*" * 20)
    print(ans)

    results.append([quest_id, ans])
    full_results.append([quest_id, question, context, ans])

with open(f"results.csv", "w", encoding="utf-8") as f:
    csv.writer(f).writerows(results)

with open("full_results.csv", "w", encoding="utf-8") as f:
    csv.writer(f).writerows(full_results)
