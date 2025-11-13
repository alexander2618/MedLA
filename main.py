import argparse
from datetime import datetime
from loguru import logger
import multiprocessing
import sys
import os
import pandas as pd
import json

import wandb
# from search.utils.utils import get_args
from random import randint

from util.agent import (
    LogicAgent,
    MedAgent_Eliminate,
    DecisionMakersAgent,
    Agent,
    parse_content
)
from util.base import base
from util.util import alpha_lower, find_answer, check_answer_same, cal_acc
from util.prompt import baseline_template, EXCEPTION_PROMPT, simple_answer_constrain_prompt


def agents_talk(agent_dict, index, retry_count=[0]):

    last_ans_reason_dict = {}
    ans_dict_angent = {}
    for agent_name, agent in agent_dict.items():
        
        if agent.ans in ans_dict_angent:
            ans_dict_angent[agent.ans].append(agent_name)
        else:
            ans_dict_angent[agent.ans] = [agent_name]
        
        last_ans_reason_dict[agent_name] = {
            'name': agent.agent_name, 
            'ans': agent.ans, 
            'reason': agent.reason + 'logic_report' + agent.logic_report}
    
    random_select_agent = []
    for ans, agent_name_list in ans_dict_angent.items():
        if len(agent_name_list) > 1:
            select_agent_index = randint(0, len(agent_name_list) - 1)
            random_select_agent.append(agent_name_list[select_agent_index])
        else: 
            random_select_agent.append(agent_name_list[0])
    
        
    # import pdb; pdb.set_trace()
    agent_dict_new = {}
    for agent_name in random_select_agent:
        agent_dict_new[agent_name] = agent_dict[agent_name]
    
    for agent_name, agent in agent_dict_new.items():
        # agent = agent_dict[agent_name]
        agent_ans = agent.ans
        ans, reason_str = agent.rebuttal(last_ans_reason_dict,retry_count=retry_count)
        logger.warning(f"problem: {index}: discussion rebuttal {agent_name}:{agent_ans}->{ans}")
    return agent_dict_new


def save_log(
    final_decision, label, question, log_list, data_index, output_file, debug=False,retry_count=0,retry_pro=0,make_decision=0,make_decision_num=[0]):
    json.dump(
        {
            "final_decision": final_decision,
            "label": label,
            "question": question,
        },
        open(output_file, "+a"),
    )
    make_decision_num[0] += make_decision
    num_right, numn_total, acc = cal_acc(open(output_file, "r").readline())
    right_str = "Right" if final_decision == label else "Wrong"
    
    log = {
            "accuracy": acc,
            "right_bool": int(final_decision == label),
            "num_right": num_right,
            "numn_total": numn_total,
            "talkTimes": len(log_list),
            "data_index": data_index,
            "retry_count": retry_count,
            "retry_pro": retry_pro/numn_total,
            "make_decision": make_decision,
            "make_decision_num": make_decision_num[0]/numn_total,
    }
    
    if data_index < 10 or final_decision != label:
        log.update({
            f"{right_str}/talkTimes{len(log_list)}/log_{data_index}_predict{final_decision}_label{label}": pd.DataFrame(
                log_list
            ),
        })
    
    wandb.log(
        log
    )


def run_single_exp(question, fewshots_data, args, label, problem_index, output_path,retry_all=[0],make_decision_num=[0]):
    retry_count=[ 0]
    log_list = []
    final_decision = None

    logicagent = LogicAgent(
        args=args,
        instruction='you are a logic agent.',
        agent_name='LogicAgent',
        logger=log_list,
    )

    agents_data = []
    for index_agent, agent in enumerate(fewshots_data):
        agents_data.append(f"Agent {index_agent}")

    agent_dict = {}
    for index_agent, agent in enumerate(agents_data):
        agent_role = agent
        agent_name = f"[{index_agent}]{alpha_lower(agent_role)}"
        inst_prompt = f"you are {agent_name}. "
        
        agent_new = MedAgent_Eliminate(
            args=args,
            instruction=inst_prompt,
            agent_name=agent_name,
            logger=log_list,
        )
        # logging.info(agent_new.temp)
        agent_dict[agent_name] = agent_new

    for agent_idx, agent_name in enumerate(agent_dict.keys()):
        agent_current = agent_dict[agent_name]
        agent_current.answer_question(question, fewshots_data[agent_idx], baseline_bool=args.baseline_bool,retry_count=retry_count)
        logic_report = logicagent.check_logic(
            agent_current.reason,retry_count=retry_count
        )
        agent_current.logic_report = logic_report
        
        logger.warning(
            f"problem: {problem_index}: first ans, {agent_name}({agent_current.ans})"
        )
    num_rounds = args.num_rounds
    for round_n in range(num_rounds):
        ans_list_c = [agent_dict[agent_name].ans for agent_name in agent_dict.keys()]
        bool_ans_same = check_answer_same(ans_list_c)

        if not bool_ans_same:
            agent_dict = agents_talk(agent_dict, problem_index,retry_count=retry_count)
        else:
            break
        logger.warning(
            f"problem: {problem_index}: discusstion, Round {round_n}"
            + f"({[agent_dict[agent_name].ans for agent_name in agent_dict.keys()]})"
        )
    making_decision = 0
    ans_list_c = [agent_dict[agent_name].ans for agent_name in agent_dict.keys()]
    bool_ans_same = check_answer_same(ans_list_c)
    if not bool_ans_same:
        moderator = DecisionMakersAgent(
            args=args,
            logger=log_list,
        )
        final_decision = moderator.make_decision(agent_dict, question, num_rounds,retry_count=retry_count)
        making_decision = 1
    else:
        final_decision = ans_list_c[0]
    logger.warning(f"problem: {problem_index}: final_decision {final_decision}")
    retry_all[0]+= retry_count[0]
    save_log(
        final_decision,
        label,
        question,
        log_list,
        problem_index,
        output_path,
        args.debug,
        retry_count=retry_count[0],
        retry_pro=retry_all[0],
        make_decision=making_decision,
        make_decision_num=make_decision_num,
    )

    return log_list, final_decision

def run_single_basiline(question, fewshots_data, args, label, problem_index, output_path,retry_all=[0],make_decision_num=[0]):
    log_list = []
    agent = Agent(
        "You are a medical expert. Answer the question based on the knowledge you have.",
        engine=args.model,
        llm_name=args.llm_name,
        agent_name='BaselineAgent',
        logger=log_list,
        port=args.port,
    )
    
    message = baseline_template.format(
        question=question[0],
        option=question[1],
    )
    retry_count = 0
    while 1:
        try:
            if retry_count > 3:
                raise Exception("Retry limit reached. Exiting...")
            opinion = agent.chat(message)
            final_decision = parse_content(opinion)
            break
        except (ValueError, IndexError) as e:
            logger.warning(
                f"{agent.agent_name} retry {retry_count}: parse error - {str(e)}; opinion: {opinion}"
            )
            message = EXCEPTION_PROMPT + simple_answer_constrain_prompt
            retry_count += 1
    
    save_log(
        final_decision,
        label,
        question,
        log_list,
        problem_index,
        output_path,
        args.debug,
        retry_count=0,
        retry_pro=0,
        make_decision=0,
        make_decision_num=[0],
    )


def cot_rag(args, wandb_logger):
    instance = base(args, wandb_logger, "test")
    if args.baseline_bool:
        instance.invoke(run_single_basiline)
    else:
        instance.invoke(run_single_exp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="medddx_expert", choices=[
        'medqa_us', 'mmlu', 'bioasq',
        "medddx_basic", "medddx_inter", "medddx_expert",])
    parser.add_argument(
        "--model",
        type=str,
        default="vllm",
        choices=[
            "vllm",
            "OLLAMA",
            "siliconflow",
            "gpt-3.5",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-11-20",
            "gpt-4o-mini-2024-07-18",
            "o3-mini",
            "glm-4-plus",
            "glm-4-flash",
            "deepseek-reasoner",
            "deepseek-chat",
            "deepseek-r1",
        ],
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="CAIRI-LLM",
        choices=[
            "CAIRI-LLM",
            "CAIRI-LLM-reasoner",
            "CAIRI-VLM",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "qwq10k",
            "gpt-4o-mini-2024-07-18",
            "deepseek-chat",
            "deepseek-reasoner",
            "gpt-4o-2024-11-20",
            "DSR1_10k70b",
            "llama3370B13K",
            "llama3370B10K",
            "llama3.3_8192",
            "gpt-4o-mini",
            "llama3.3_4096",
            "DS-r1:70b",
            "qwen2.5:72b_6000",
            "DS-r1:70b",
            "llama3.3",
        ],
    )
    parser.add_argument("--few_shots", type=int, default=1)
    parser.add_argument("--knowledges", type=int, default=0)
    parser.add_argument("--db_size", type=int, default=100)
    parser.add_argument("--num_agents", type=int, default=17)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--temp", type=float, default=0.7)

    parser.add_argument("--num_times", type=int, default=1)
    parser.add_argument("--rag", type=str, default=None, choices=["nomic-embed-text", "mxbai-embed-large"])
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--baseline_bool", type=bool, default=False)

    parser.add_argument("--url", type=str, default="http://127.0.0.1:7777")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--num_threads", type=int, default=24)
    parser.add_argument(
        "--tag", type=str, default="recover"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb", type=int, default=os.environ.get("ADAMA_PORT", "Adama"))
    
    args = parser.parse_args()

    args.num_threads = int(os.environ.get("ADAMA_NUM_THREADS", args.num_threads))

    if args.debug:
        args.num_samples = 1
        args.times = 1
        args.num_threads = 1

    return args


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    
    # print the command line arguments
    print("Command line arguments:", sys.argv)
    print("Number of arguments:", len(sys.argv))
    print("Arguments:", sys.argv[1:])

    args = get_args()
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb_logger = wandb.init(
        project=args.wandb,
        name=f"{args.model}_{args.dataset}_{args.tag}_{time_str}",
        config=args,
    )

    cot_rag(args, wandb_logger)

    wandb.finish()
