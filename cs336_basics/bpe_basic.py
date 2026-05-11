from collections import Counter

def main(text: str):
    # pre_tokenization
    pre_tokens_list = text.split()
    tuples_list = []
    for token in pre_tokens_list:
        tuples_list.append((char for char in token))
        
    frequency_dict_origin = Counter(pre_tokens_list)
    frequency_dict = Counter()
    # scan_begin   # 为什么这里vscode debugger显示tuples_list内部是一个个字符串元素而不是元组？
    for tuple in tuples_list:
        for i in range(len(tuple)):
            if i > 0:
                frequency_dict[tuple[i-1] + tuple[i]] += frequency_dict_origin[tuple]

    return frequency_dict
    


if __name__ == "__main__":
    main('abc ccc')
