from datetime import datetime
import json
from joblib import Parallel, delayed
from loguru import logger
import os
from tqdm import tqdm


from .rag import QAEmbeddingNewDB
from .load_data import create_question
from .load_data import load_data_medqa, load_data_medqa_us, load_data_pubmedqa, load_data_mmlu, load_data_bioasq, load_data_medddx


class base:
    def __init__(self, args, wandb_logger, phase="test"):
        self.args = args
        self.phase = phase
        self.acc = None
        self.qDB = QAEmbeddingNewDB(
            db_size=args.db_size,
            model=args.rag,
        )
        self.wandb_logger = wandb_logger
        logger.info(str(self.args))

        self.preprocess()

    def preprocess(self):
        if self.args.dataset == "sub" or self.args.dataset == "medqa" or self.args.dataset == 'subzl':
            test_qa, examplers = load_data_medqa(self.phase, self.args.seed)

        elif self.args.dataset == "medqa_us":
            test_qa, examplers = load_data_medqa_us()
        
        elif self.args.dataset == "pubmedqa":
            test_qa, examplers = load_data_pubmedqa()

        elif self.args.dataset == "mmlu":
            test_qa, examplers = load_data_mmlu()

        elif self.args.dataset == "bioasq":
            test_qa, examplers = load_data_bioasq()

        elif self.args.dataset == "medddx_basic":
            test_qa, examplers = load_data_medddx("basic")
            
        elif self.args.dataset == "medddx_inter":
            test_qa, examplers = load_data_medddx("inter")
            
        elif self.args.dataset == "medddx_expert":
            test_qa, examplers = load_data_medddx("hard")

        if self.args.num_samples != -1:
            test_qa = test_qa[: self.args.num_samples]
        logger.info(f"Loading {len(test_qa)} samples in total")

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.args.model}_{self.args.dataset}_{len(test_qa) if self.args.num_samples == -1 else self.args.num_samples}_{self.args.tag}_{current_time}.json"
        output_path = os.path.join(
            os.getcwd(), "output", current_time.split("_")[0], self.args.dataset
        )
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = os.path.join(output_path, file_name)
        logger.info(f"Output file: {output_file}")
        # 如果文件存在，加载已经记录的结果，避免重复计算
        # recorded_results["result"] = []

        self.dataset = test_qa
        self.examplers = examplers
        self.output_file = output_file


    def invoke(self, run_single_exp):
        retry_all=[0]
        make_decision_num = [0]
        question_list = []
        few_shot_exp_list = []
        for idx in range(len(self.dataset)):
            question, options = create_question(self.dataset[idx], self.args.dataset)
            get_few_shot_data = self.qDB.get_few_shot(question, self.args.few_shots, self.args.num_agents)
            question_list.append([question, options])
            few_shot_exp_list.append(get_few_shot_data)
        
        # import pdb; pdb.set_trace()
        
        if self.args.num_threads == 1:
        # for debug
            for idx in range(len(self.dataset)):
                run_single_exp(
                    question_list[idx], 
                    few_shot_exp_list[idx], 
                    self.args,
                    self.dataset[idx]["answer_idx"],
                    idx,
                    self.output_file, 
                )
        else:        
            results = Parallel(n_jobs=self.args.num_threads, backend="threading")(
                delayed(run_single_exp)(
                    question_list[idx], 
                    few_shot_exp_list[idx], 
                    self.args,
                    self.dataset[idx]["answer_idx"],
                    idx,
                    self.output_file, 
                    retry_all=retry_all,
                    make_decision_num=make_decision_num,
                )
                for idx in range(len(self.dataset))
            )


if __name__ == "__main__":

    # def func(xs, p_bar, queue):
    #     for x in xs:
    #         queue.put(x)
    #         p_bar.update(1)

    # instance = base(5)
    # instance.invoke(func, [111, 222, 2334, 341, 234, 1234])
    
    # a, b = load_data("sub", "test")
    # a, b = load_data("pubmedqa", "test")
    # a, b = load_data("mcqa", "test")
    
    # print(a[0], "\n----------------------------------\n", b[0])
    # print("----------------------------------")
    # print("test: ", len(a), ", example: ", len(b))
    
    pass
    