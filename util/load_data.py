import json
import jsonlines
import random
import pandas as pd
import os
from glob import glob
from base64 import b64encode
from tqdm import tqdm

import random


def create_question(sample, dataset):
    question = sample["question"]
    options = {}
    for k, v in sample["options"].items():
        # options.append("({}) {}".format(k, v))
        options[k] = v
    
    return question, options


def load_data_medqa_us():
    reader = jsonlines.open("data/MIRAGE/rawdata/medqa/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl", "r")
    test_qa = [qa for qa in reader]
    
    print("medqa_us test_qa: ", len(test_qa))
    
    return test_qa, None


def load_data_mmlu():
    options = ['A', 'B', 'C', 'D']
    test_qa = []
    
    base_path = "data/MIRAGE/rawdata/mmlu/data/test"
    test_data_df = None
    for file in glob(f"{base_path}/*.csv"):
        df = pd.read_csv(file, header=None)
        if test_data_df is None:
            test_data_df = df
        else:
            test_data_df = pd.concat([test_data_df, df])
    test_data_df = test_data_df.reset_index(drop=True)
    
    test_qa = []
    for idx, row in test_data_df.iterrows():
        test_qa.append({
            "idx": idx,
            "question": row[0],
            "options": {"A": row[1], "B": row[2], "C": row[3], "D": row[4]},
            "answer": row[options.index(row[5]) + 1],
            "answer_idx": row[5],
        })
        
    print("mmlu test_qa: ", len(test_qa))

    return test_qa, None


def load_data_bioasq():

    options = {"A": "yes", "B": "no"}
    questions = []
    for file in glob("data/MIRAGE/rawdata/bioasq/Task*BGoldenEnriched/*.json"):
        questions.extend(json.load(open(file, "r", encoding="utf-8"))['questions'])
        
    questions = [question for question in questions if "exact_answer" in question and isinstance(question['exact_answer'], str) and question['exact_answer'].lower() in ["yes", "no"]]

    test_qa = []
    for idx, question in enumerate(questions):
        test_qa.append({
            "idx": idx,
            "question": question['body'],
            "options": options,
            "answer": question['exact_answer'],
            "answer_idx": "A" if question['exact_answer'].lower() == "yes" else "B",
        })
        
    print("bioasq test_qa: ", len(test_qa))

    return test_qa, None


def load_data_medddx(seed):
    chars=[]
    for i in range(65,91):
        chars.append(chr(i))
    def get_options(query):
        options = ("A: " + query.split("\nA: ")[1]).split("\n")
        try:
            output = {chars[idx]: option.split(": ")[1] for idx, option in enumerate(options) if len(option) > 0}
        except:
            print(query.split("\nA: ")[0], options)
        return output
    
    benchmark = json.load(open("data/medddx/MedDDx.json"))
    if seed == "hard":
        # hard
        datas = [b for b in benchmark if b['sim_level_std'] < 0.02]
        test_qa = [{
            "idx": idx,
            "question": data['query'].split("\nA: ")[0],
            "options": get_options(data['query']),
            "answer_idx": data['answer'],
            } for idx, data in enumerate(datas)]
        
        for data, qa in zip(datas, test_qa):
            qa['answer'] = qa["options"][data['answer']]
        
    if seed == "inter":    
        # intermidiate
        datas = [b for b in benchmark if b['sim_level_std'] > 0.02 and b['sim_level_std'] < 0.04]
        test_qa = [{
            "idx": idx,
            "question": data['query'].split("\n")[0],
            "options": get_options(data['query']),
            "answer_idx": data['answer'],
            } for idx, data in enumerate(datas)]
        
        for data, qa in zip(datas, test_qa):
            qa['answer'] = qa["options"][data['answer']]
        
    if seed == "basic":
        # basic
        datas = [b for b in benchmark if b['sim_level_std'] > 0.04]
        test_qa = [{
            "idx": idx,
            "question": data['query'].split("\n")[0],
            "options": get_options(data['query']),
            "answer_idx": data['answer'],
            } for idx, data in enumerate(datas)]
        
        for data, qa in zip(datas, test_qa):
            qa['answer'] = qa["options"][data['answer']]
            
    print(f"medddx {seed} test_qa: ", len(test_qa))

    return test_qa, None


if __name__ == "__main__":
    load_data_medqa_us()
    load_data_mmlu()
    load_data_bioasq()
    
    load_data_medddx("hard")
    load_data_medddx("inter")
    load_data_medddx("basic")
    