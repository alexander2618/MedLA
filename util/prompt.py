from langchain.prompts import PromptTemplate


# __all__ = [
#     "EXCEPTION_PROMPT",
#     "medagent_answer_template",
#     "medagent_answer_rule_prompt",
#     "medagent_rebuttal_template",
#     "logic_answer_template",
#     "eliminate_answer_prompt",
#     "eliminate_answer_template",
#     "eliminate_rebuttal_template",
#     "compair_answer_template",
#     "compair_answer_rule_prompt",
#     "compair_rebuttal_template",
#     "exclusive_answer_template",
#     "exclusive_answer_rule_prompt",
#     "exclusive_rebuttal_template",
#     "decision_maker_template",
# ]

EXCEPTION_PROMPT ="Pay attention to the format and answer again. Follow the format below: \n<Answer>Answer: A/B/C/D/E</Answer>\n<Logic> </Logic>\n.(don't forget the <Logic> </Logic> and <Answer> </Answer> tags.)"

simple_answer_constrain_prompt = '''Follow the format below: 
<Answer>Answer: A/B/C/D/E</Answer>.
(don't forget the <Answer> </Answer> tags.)'''

# logic agent prompt
logic_answer_prompt = """reasoning passage:{logic}

Please perform the following tasks:
1. [Step-by-step Reasoning Reconstruction]  
Break down the entire reasoning process step-by-step. Identify:
- The subject (who or what the step is about)
- The object (the outcome, result, or related concept)
- The logical relationship (e.g., cause-effect, association, inference)
- The logical strength (Strong / Moderate / Weak)
- Whether the step is flawed (Yes / No)
- If flawed, specify the error type (Factual / Logical / Conceptual)

2. [TSV Table Output]  
Provide your analysis in TSV table format with the following columns:
Step, Subject, Object, Logical Relationship, Reasoning Text, Credibility, Error (Yes/No), Error Type, Suggested Correction

3. [Optional Full Report]  
After the TSV table, you may include a full written report including:
- Summary of reasoning
- Overall assessment of reasoning strength
- Suggestions for correction

Example of the table format:
Step\tSubject\tObject\tLogical Relationship\tReasoning Text\tCredibility\tError (Yes/No)\tError Type\tSuggested Correction  
Fever\tBacterial Infection\tSymptom-to-cause\tThe presence of fever indicates a bacterial infection\tWeak\tYes\tFactual\tClarify that fever is non-specific and may be caused by viral or bacterial infection  
...

Make sure your TSV-style output is accurate, logically sound, and grounded in standard medical knowledge.
Just return this Tsv form."""

logic_answer_constrain_prompt = '''Make sure your TSV-style output is accurate, logically sound, and grounded in standard medical knowledge.
Just return this Tsv form.

Example of the table format:
Step\tSubject\tObject\tLogical Relationship\tReasoning Text\tCredibility\tError (Yes/No)\tError Type\tSuggested Correction  
Fever\tBacterial Infection\tSymptom-to-cause\tThe presence of fever indicates a bacterial infection\tWeak\tYes\tFactual\tClarify that fever is non-specific and may be caused by viral or bacterial infection  
...'''

logic_answer_template = PromptTemplate.from_template(logic_answer_prompt)


# Eliminate agent
eliminate_answer_prompt = """{input}
Question: {question}
Option: {option}
Help me eliminate the least likely option. 
Then explain in one sentence the key reason for such a judgment.
Please follow the structured reasoning steps below and provide the final answer clearly:
Step 1: Information Extraction  
List clearly each important piece of information provided in the question:
- Information 1: ...
- Information 2: ...
- Information 3: ...
(Continue as needed)
Step 2: Background Knowledge  
Clearly state relevant background knowledge or general principles necessary for solving this problem:
- Knowledge 1: ...
- Knowledge 2: ...
- Knowledge 3: ...
(Continue as needed)
Step 3: Reasoning Process (Subject-Predicate-Object format)  
Perform logical reasoning step-by-step, explicitly indicating each reasoning step as a Subject-Predicate-Object triple, along with a brief explanation of the logical relationship:
Reasoning Step 1:
- Subject: ...
- Predicate (relation): ...
- Object: ...
- Explanation: ...
Reasoning Step 2:
- Subject: ...
- Predicate (relation): ...
- Object: ...
- Explanation: ...
Reasoning Step 3:
- Subject: ...
- Predicate (relation): ...
- Object: ...
- Explanation: ...
(Continue additional reasoning steps as necessary)
- Object: The entity or concept that receives the action or judgment.
----
Follow the format below: 
<Eliminate>Answer: {option_str}</Eliminate> 
Reason: *** 
(don't forget the <Eliminate> </Eliminate> tags.)"""

eliminate_answer_template = PromptTemplate.from_template(eliminate_answer_prompt).partial(
    # constrain=eliminate_answer_constrain_prompt.format()
)

eliminate_rebuttal_prompt = '''Question:{question}
Your opinion is: {ans}. Your reasoning is: {reason}
Please communicate with multiple other logical perspectives.
{opinions}
First answer which should be the correct option.
Then, please summarize each person's logic, and if you find them to be factually incorrect, and logically incorrect, please list them below and give reasons.
-----
Follow the format below: 
<Answer>Answer: {option_str}</Answer>.
(don't forget the <Answer> </Answer> tags.)'''

eliminate_rebuttal_template = PromptTemplate.from_template(eliminate_rebuttal_prompt).partial(
    # constrain=simple_answer_constrain_prompt,
)


eliminate_prompt = '''Follow the format below: 
<Eliminate>Answer: {option}</Eliminate> 
Reason: *** 
(don't forget the <Eliminate> </Eliminate> tags.)'''

answer_prompt = '''Follow the format below: 
<Answer>Answer: {option}</Answer>.
(don't forget the <Answer> </Answer> tags.)'''

# Decision maker Agent
decision_maker_prompt = '''You are the ultimate medical decision-maker, responsible for reviewing all opinions from various medical experts and making the final decision.
The question is: {question}
The team members had {num_rounds} rounds of discussion, and the opinions from each round are as follows:
{opinions}
Please review the opinions of each expert and provide a final answer based on the medical knowledge question.'''

decision_maker_template = PromptTemplate.from_template(decision_maker_prompt)

baseline_prompt = '''Answer my questions as requested without any other extraneous content. You must choose one of these options as the answer.
Question: {question}
Option: {option}
Follow the format below: 
<Answer>Answer: A/B/C/D/E</Answer>.
(don't forget the <Answer> </Answer> tags.)'''

baseline_template = PromptTemplate.from_template(baseline_prompt)


if __name__ == "__main__":
    print(logic_answer_template.format(
        logic="The patient has a fever, which indicates a bacterial infection.",
    ))
