from loguru import logger
import numpy as np
import os
import random


class QADB:
    def __init__(
        self,
        db_size=200,
    ):

        path = f'data/subzl/logic/train6prompt_compair_logic.jsonl'
        with open(path, 'r') as f:
            lines_compair = f.readlines()
        path = f'data/subzl/logic/train6prompt_Eliminate_logic.jsonl'
        with open(path, 'r') as f:
            lines_Eliminate = f.readlines()
        path = f'data/subzl/logic/train6prompt_mutually_exclusive_logic.jsonl'
        with open(path, 'r') as f:
            lines_mutually = f.readlines()

        self.examples = lines_compair+lines_Eliminate+lines_mutually
        
        random.seed(999)
        random.shuffle(self.examples)

        if db_size != -1 and db_size < len(self.examples):
            self.examples =  self.examples[:db_size]
        logger.info(f"examples number: {len(self.examples)}")
        
    def get_few_shot(self, k: int = 1, num_agents: int = 1):
        
        if k == 0:
            return [(None, None)] * num_agents

        output = []
        # for predict_label in sorted_cluster_indices:
        for _ in range(num_agents):

            all_indices_cluster = np.arange(len(self.examples))
            indices = np.random.choice(len(all_indices_cluster), k, replace=False)

            predict_all_index = all_indices_cluster[indices]

            few_shot_str = ''
            for i, idx in enumerate(predict_all_index):
                predict_labels_str = self.examples[idx]
                question_str_info = predict_labels_str.split('\\"question\\"')[1].split('logic')[0]
                question_str_info = question_str_info.replace('\\', '')
                logic_str_info = predict_labels_str.split('logic:')[1]
                logic_str_info = logic_str_info.replace('\\n', '\n')
                
                if 'prompt_Eliminate_logic' in predict_labels_str:
                    logic_str_info_log_info = 'prompt_Eliminate_logic'
                elif 'prompt_compair_logic' in predict_labels_str:
                    logic_str_info_log_info = 'prompt_compair_logic'
                elif 'prompt_mutually_exclusive_logic' in predict_labels_str:
                    logic_str_info_log_info = 'prompt_mutually_exclusive_logic'
                else:
                    os.error('No logic type')
                few_shot_str += f'[Example {i}] Question: {question_str_info} \n {logic_str_info_log_info}: {logic_str_info} \n\n'

            output.append((few_shot_str, None))
        
        # import pdb; pdb.set_trace()
        return output


if __name__ == "__main__":
    db = QADB()
    ans = db.get_few_shot('What is the mechanism of action of aspirin?', 3, 3, 5)
    print(ans[0])
    print(len(ans))
    print(ans[1])
    ans = db.get_few_shot('What is the mechanism of action of aspirin?', 0, 3, 5)
    print(ans[0])
    print(len(ans))
    print(ans[1])
    ans = db.get_few_shot('What is the mechanism of action of aspirin?', 3, 3, 0)
    print(ans[0])
    print(len(ans))
    print(ans[1])
    
    # instance = vdb()
    # print(instance.invoke(['What is the mechanism of action of aspirin?', 'What is the mechanism of action of ibuprofen?']))
    # instance.add(
    #     ['3', '4'],
    #     ['What is the mechanism of action of aspirin???', 'What is the mechanism of action of ibuprofen???????'],
    # )
    # print(len(instance))
    
    # print(instance.query('What is aspirin?'))
    
    # print(list(range(10)))
    