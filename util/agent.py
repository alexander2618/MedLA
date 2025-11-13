from datetime import datetime
import uuid
import io
from loguru import logger
import random
import pandas as pd
from pathlib import Path
import json
from time import sleep
from openai import OpenAI
from zhipuai import ZhipuAI
import os

from .prompt import *


def csv_to_pandas(csv_string):
    if csv_string.startswith('```tsv'):
        # 将markdown格式的tsv转成dataframe
        clean_data = csv_string.strip()
        for mark in ['```tsv', '```csv', '```']:
            clean_data = clean_data.replace(mark, '')
        pandas_df = pd.read_csv(
            io.StringIO(clean_data.strip()),
            sep="\t",
            engine='python',
            skipinitialspace=True,
            dtype=str  # 初始读取为字符串类型
        )
    else:

        list_zzl = csv_string.split("\n")

        table = []
        for i in range(len(list_zzl)):
            if len(list_zzl[i]) > 0:
                table.append(list_zzl[i].split("\t"))
        pandas_df = pd.DataFrame(table[1:], columns=table[0])

    return pandas_df


def find_answer(str_ans, ans_list=["A", "B", "C", "D", "E"]):

    result_out_list = []
    for ans in ans_list:
        if ans in str_ans:
            result_out_list.append(ans)

    if len(result_out_list) == 1:
        return result_out_list[0]
    else:
        # rise error
        # raise ValueError(f"find_answer cannot find ans: {str_ans}")
        return ""


def check_answer_keyword(str_all, check_str, tag):

    checked_count = 0
    if f"<{tag}>Answer: {check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"<{tag}> Answer: {check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"<{tag}>Option: {check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"<{tag}> Option: {check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"<{tag}>Option {check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"<{tag}> Option {check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"<{tag}>{check_str}</{tag}>".lower() in str_all.lower():
        checked_count += 1
    if f"<{tag}> {check_str} </{tag}>".lower() in str_all.lower():
        checked_count += 1
    if f"\n {tag}:{check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"\n {tag}: {check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"{tag}: {check_str}</{tag}>".lower() in str_all.lower():
        checked_count += 1
    if f"{tag}:{check_str}</{tag}>".lower() in str_all.lower():
        checked_count += 1

    if checked_count > 0:
        return True
    else:
        return False


def parse_content(content, tag="Answer") -> str:
    ans = ""
    if check_answer_keyword(content, "A", tag):
        ans += "A"
    elif check_answer_keyword(content, "B", tag):
        ans += "B"
    elif check_answer_keyword(content, "C", tag):
        ans += "C"
    elif check_answer_keyword(content, "D", tag):
        ans += "D"
    elif check_answer_keyword(content, "E", tag):
        ans += "E"

    if len(ans) == 1:
        return ans
    elif len(ans) == 0:
        # import pdb; pdb.set_trace()
        raise ValueError(f"Unable to find label <{tag}> in text ")
    else:
        raise ValueError(f"Finding multiple labels in text: {ans}")


def parse_after_colon(text: str) -> str:
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text.strip()


class AgentHelpler:
    def __init__(self, engine, agent_name, url="http://127.0.0.1:7778") -> None:
        self.default_temp = None
        self.agent_name = agent_name + str(uuid.uuid4())[:10]
        self.api_key = os.getenv("API_KEY", "")
        if engine != "openai" and self.api_key == "":
            raise ValueError("API_KEY environment variable is not set.")
        # self.encoding_tiktoken = tiktoken.encoding_for_model("gpt-4")
        if engine == "openai":
            self.client = OpenAI(
            )

        elif engine in ["deepseek"]:
            # https://api.deepseek.com
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=f"https://api.deepseek.com/v1",
            )

        elif engine in ["vllm"]:
            if url.ednswith("/v1"):
                url = url[:-3]
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=f"{url}/v1",
            )

        elif engine in ["siliconflow"]:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.siliconflow.cn/v1",
            )

        elif engine in ["zhipuai"]:
            self.client = ZhipuAI(
                api_key=self.api_key
            )

        else:
            raise ValueError(f"Unsupported model: {engine}")
        
    def invoke(self, llm_name, messages, temperature=None):
        retry = 3
        # logger.warning(temperature)

        if temperature is None:
            temperature = AgentHelpler.temp
        while retry > 0:
            response = self.client.chat.completions.create(
                model=llm_name,
                messages=messages,
                temperature=temperature,
                max_tokens=8000
                )
            # logger.error(temperature)
            out = response.choices[0].message.content
            
            if out is not None:
                break
            sleep(10)
            retry -= 1
            logger.warning(f"Retrying... {retry} attempts left.")
        else:
            logger.error("Failed to get a response after multiple attempts.")
            raise ValueError("No response from the model.")

        if "</think>" in out:
            out = out.split("</think>")[1]
            
        return out


class Agent(AgentHelpler):
    def __init__(
        self,
        args,
        system_instruction,
        retry_times=3,
        agent_name=None,
        logger: list = None,
    ):
        self.args = args
        self.engine = args.model
        self.llm_name = args.llm_name
        self.url = args.url

        self.logger = logger
        self.retry_times = retry_times
        
        self.ans_list = []
        self.reason_list = []
        self.challence_info_list = []
        self.system_instruction = system_instruction
        self.history = {"with_history": [{"role": "system", "content": system_instruction}], "without_history": []}
        self.messages = [
            {"role": "system", "content": system_instruction},
        ]
        super().__init__(self.engine, agent_name=agent_name, url=self.url)

    def __del__(self):
        if self.args.debug:
            message_file = Path(__file__).parent.joinpath("output", datetime.now().strftime('%Y%m%d'), self.args.dataset, f"debug_{self.agent_name}.json")
            with open(message_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=4)
        

    def chat_without_history(self, message, system_instruction=None, temperature=None):
        temperature = self.args.temp if temperature is None else temperature
            
        message_new = [
            {"role": "system", "content": self.system_instruction if system_instruction is None else system_instruction},
            {"role": "user", "content": message},
        ]
        
        out = self.invoke(self.llm_name, message_new, temperature)
        message_new.append({"role": "assistant", "content": out})
        self.history['without_history'].append(message_new)

        return out

    def chat(self, message, img_path=None, chat_mode=True, temperature=None):
        temperature = self.args.temp if temperature is None else temperature

        self.messages.append({"role": "user", "content": message})
        # import pdb; pdb.set_trace()
        
        out = self.invoke(self.llm_name, self.messages, temperature)    

        self.messages.append({"role": "assistant", "content": out})
        
        self.history['with_history'].extend([{"role": "user", "content": message},{"role": "assistant", "content": out}])

        return out

    def add_log(self, agent_name, step, input, info, answer):
        self.logger.append(
            {
                "agent_name": agent_name,
                "step": step,
                "input": input,
                "info": info,
                "answer": answer,
            }
        )

class LogicAgent(Agent):
    def __init__(
        self,
        args,
        instruction,
        agent_name=None,
        logger=None,
        retry_times=10,
    ):
        agent_name = "MedAgent_" + agent_name
        
        instruction = f"You are a professional medical logic checker. Your task is to analyze a given medical reasoning passage and identify each reasoning step, its logical structure, and potential issues. In addition to a written evaluation, you must output a TSV-style table summarizing each reasoning step.\n{logic_answer_constrain_prompt}"

        super().__init__(
            args=args,
            system_instruction=instruction,
            retry_times=retry_times,
            agent_name =agent_name, 
            logger=logger, 
            )

    def check_logic(self, logic, baseline_bool=False, retry_count=[0]):

        message = logic_answer_template.format(
            logic=logic,
        )
        
        opinion_str = ""
        reason_str = ""
        ans = ""
        retry = 0
        while len(reason_str) < 1 and retry < self.retry_times:
            retry += 1
            try:

                opinion = self.chat_without_history(message)
 
                reason_str = opinion
            except (ValueError, IndexError) as e:
                logger.error(
                    f"{self.agent_name} retry {retry}: parse error - {str(e)}; opinion: {opinion}"
                )
                # message = EXCEPTION_PROMPT + logic_answer_constrain_prompt + "\n" + message
                retry_count[0]+=1
                continue
            except Exception as e:
                logger.error(f"{self.agent_name} retry {retry}: unknow error - {str(e)} ")
                break

        self.ans = ans
        self.reason = reason_str
        self.ans_list.append(self.ans)
        self.reason_list.append(self.reason)

        self.add_log(self.agent_name, "check_logic", message, reason_str, ans)
        return reason_str


class MedAgent_Eliminate(Agent):
    def __init__(
        self,
        args,
        instruction,
        agent_name=None,
        logger=None,
        retry_times=10,
    ):
        agent_name = "MedAgent_" + agent_name
        
        super().__init__(
            args=args,
            system_instruction=instruction,
            retry_times=retry_times,
            agent_name=agent_name, 
            logger=logger, 
        )

    def Eliminate_one_answer(self, question, option, examples):
        option_keys = list(option.keys())  
        random.shuffle(option_keys)  
        option_str = '/'.join(option_keys)
        
        k1 = self.args.few_shots
        k2 = self.args.knowledges
        
        if k1 == 0 and k2 == 0:
            input_str = ""
        else:
            examples, knowledge = examples
            input_str = "Based the following "
        
            if k1 * k2 > 0:
                input_str += f"examples and knowledge, return the answer to the medical question.\nExamples are as follows:\n{examples}\nKnowledge are as follows:\n{knowledge}"
            elif k1 > 0:
                input_str += f"examples, return the answer to the medical question.\nExamples are as follows:\n{examples}"
            elif k2 > 0:
                input_str += f"knowledge, return the answer to the medical question.\nKnowledge are as follows:\n{knowledge}"
        
        message = eliminate_answer_template.format(
            input=input_str,
            question=question,
            option=str(option),
            option_str=option_str
        )
        opinion = self.chat_without_history(message, self.system_instruction + f"Help me eliminate the least likely option. \nMake sure {eliminate_prompt}")
        
        return message, opinion

    def answer_question(
        self, question_option, examples, baseline_bool=False,retry_count=[0]
    ):

        self.examples = examples
        self.question = question_option

        question, option_meta = question_option
        option = option_meta.copy()
        reason_dict = {}
        reason_str = ""
        message_str = ""
        retry_times = 0
        while retry_times < self.retry_times and len(option) > 1:
            try:
                # print("answer_question")
                message, reason = self.Eliminate_one_answer(
                    question, option, examples
                )
                Eliminate_ans = parse_content(reason, tag='Eliminate')
                reason_dict[Eliminate_ans] = reason
                ans_cont = option.pop(Eliminate_ans)
                message_str += message + "\n--------------\n"
                reason_str += reason + "\n--------------\n"
                logger.warning(f"remove ({Eliminate_ans}), {ans_cont}")
            except (ValueError, IndexError, KeyError) as e:
                retry_times += 1
                retry_count[0]+=1
                logger.error(reason)
                logger.error(f"{self.agent_name} retry: {retry_times}, parse error - {str(e)}; ")

        self.ans = list(option.keys())[-1]
        self.reason = reason_str

        self.ans_list.append(self.ans)
        self.reason_list.append(self.reason)

        self.add_log(
            self.agent_name, "answer_question", message_str, reason_str, self.ans
        )
        return self.ans

    def rebuttal(self, agent_dict, max_retries=5,retry_count=[0]):
        opinions = "---\n"
        for agent_name2, agent_current2 in agent_dict.items():
            if agent_dict[agent_name2]["name"] != self.agent_name:
                other_agent = agent_current2
                opinions += f"The opinion of {agent_name2} is {other_agent['ans']}, and the reasoning is: {other_agent['reason']}.\n ---\n"

        option_str = "/".join(set([v['ans'] for v in agent_dict.values()]))
        message = eliminate_rebuttal_template.format(
            question=self.question,
            ans=self.ans,
            reason=self.reason,
            opinions=opinions,
            option_str=option_str
        )
        
        opinion_str = ""
        reason_str = ""
        ans = ""
        retry = 0
        while len(ans) < 1 and retry < max_retries:
            retry += 1
            try:
                opinion = self.chat_without_history(message, self.system_instruction + f"Analysis other peron's logic and find which answer should be the correct option\nMake sure {answer_prompt}")

                ans = parse_content(opinion)
                reason_str = opinion
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"{self.agent_name} retry {retry}: parse error - {str(e)}; opinion: {opinion}"
                )
                message = EXCEPTION_PROMPT + simple_answer_constrain_prompt
                retry_count[0]+=1
                continue
            except Exception as e:
                logger.error(f"{self.agent_name} retry {retry}: unknow error - {str(e)} ")
                break

        self.ans = ans
        self.reason = reason_str
        self.ans_list.append(self.ans)
        self.reason_list.append(self.reason)

        self.add_log(self.agent_name, "rebuttal", message, reason_str, ans)
        return ans, reason_str


class DecisionMakersAgent(Agent):
    def __init__(
        self,
        args,
        instruction=f"You are the ultimate medical decision-maker, responsible for reviewing all opinions from various medical experts and making the final decision.\n Make sure {simple_answer_constrain_prompt}\n",
        agent_name="DecisionMakers",
        logger=None,
        retry_times=5,
    ):
        agent_name = "DecisionMakers_" + agent_name
        self.args = args

        super().__init__(
            args=args,
            system_instruction=instruction,
            retry_times=retry_times,
            agent_name=agent_name, 
            logger=logger, 
            )

    def make_decision(self, agent_dict, question, num_rounds, retry_count=[0]):
        opinions = ""
        for i in range(num_rounds + 1):
            if i == 0:
                opinions += f"\nInitial opinions: -------\n"
            else:
                opinions += f"\nOpinions from Round {i + 1}: ------\n"
            # for agent in agent_list:
            for agent_name in agent_dict.keys():
                agent = agent_dict[agent_name]
                opinions += f"The opinion of {agent.agent_name} is: {agent.ans_list[i]}. The reason is: {agent.reason_list[i]}.\n"

        message = decision_maker_template.format(
            question=question,
            num_rounds=num_rounds,
            opinions=opinions,
        )

        retry = 0
        reason_str = ""
        opinion_str = ""
        final_decision = ""
        while len(final_decision) < 1 and retry < self.retry_times:
            retry += 1
            try:
                opinion = self.chat(message)
                extracted_opinion = parse_content(opinion)
                opinion_str = parse_after_colon(extracted_opinion)
                final_decision = find_answer(opinion_str)
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"{self.agent_name} retry {retry}: parse error - {str(e)}; opinion: {opinion}"
                )
                message = EXCEPTION_PROMPT
                retry_count[0]+=1
                continue
            except Exception as e:
                logger.error(f"{self.agent_name} retry {retry}: unknow error - {str(e)} ")
                break

            reason_str = opinion

        self.add_log(
            self.agent_name, "make_decision", message, reason_str, final_decision
        )
        return final_decision
