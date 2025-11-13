from loguru import logger

def alpha_lower(s):
    """
    Converts only English alphabetic characters in the string to lowercase.

    Args:
        s (str): The input string.

    Returns:
        str: The string with English letters converted to lowercase.
    """
    return "".join(
        [char.lower() if char.isalpha() and char.isascii() else char for char in s]
    )

def find_answer(str_ans, ans_list=["A", "B", "C", "D", "E"]):

    result_out_list = []
    for ans in ans_list:
        if ans in str_ans:
            result_out_list.append(ans)

    if len(result_out_list) == 1:
        return result_out_list[0]
    else:
        return None
    
def check_answer_same(ans_list):
    # check if all the answers are the same
    return len(set(ans_list)) == 1


def cal_acc(log_str):
    
    predict_list = []
    deci_list_raw = log_str.split('"final_decision":')
    for deci in deci_list_raw:
        ans_str = deci.split(", ")[0]
        predict_list.append(ans_str)
        
    label_list = []
    label_list_raw = log_str.split('"label":')
    for label in label_list_raw:
        label_str = label.split(", ")[0]
        label_list.append(label_str)
    
    num_right = 0
    numn_total = 0
    for i in range(1, len(predict_list)):
        # print(f"Predict: {predict_list[i]}, Label: {label_list[i]}")
        if predict_list[i] == label_list[i]:
            num_right += 1
        numn_total += 1
    
    logger.info(f"Accuracy: {num_right}/{numn_total} = {num_right/numn_total}")
    return num_right, numn_total, num_right/numn_total
