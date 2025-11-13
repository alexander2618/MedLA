def check_answer_keyword(str_all, check_str, tag):

    checked_count = 0
    if f"<{tag}>Answer: {check_str}".lower() in str_all.lower():
        checked_count += 1
    if f"<{tag}>{check_str}</{tag}>".lower() in str_all.lower():
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

opinion = f"""
<Eliminate>D/A/B/C</Eliminate> 
Reason: ***Cerulenin (B) is not typically used to treat candidiasis with vaginitis; it is primarily known for its inhibitory effects on fatty acid synthesis.***

Eliminate_one_answer

"""
Eliminate_ans = parse_content(opinion, tag='Eliminate')
print(Eliminate_ans)