import jsonlines

def main():
    path = 'data/subzl/seed_42.jsonl'
    logic_path = 'data/medqa/logic/train.jsonl'
    all_txt = open(logic_path, 'r').readlines()
    
    reader = jsonlines.open(path, "r")

    # print(test_qa[0]['question'])
    for idx, qa in enumerate(reader):
        for txt in all_txt:
            if qa['question'][:100] in txt:
                print(idx, 'found!')
            else:
                print(idx, 'not found!')
    import pdb; pdb.set_trace()
    print(all_txt[0])
    
    

if __name__ == "__main__":
    main()
