from collections import Counter

def main(text: str):
    # pre_tokenization
    pre_tokens_list = text.split()
    tuples_list = []
    for token in pre_tokens_list:
        tuples_list.append(tuple(token))  # tuple(str)返回的是逐char拆开的，一个字符一个元组内元素
    # python没有元组生成式!!下面的写法append的是一个迭代器对象
        # tuples_list.append((char for char in token))
    
    frequency_dict_origin = Counter(tuples_list)
    frequency_dict = Counter()
    # scan_begin
    for pre_token in frequency_dict_origin:
        for i, char in enumerate(pre_token):
            if i > 0:
                frequency_dict[pre_token[i - 1] + pre_token[i]] += frequency_dict_origin[pre_token]     

    return frequency_dict
    


if __name__ == "__main__":
    main('''low low low low low
lower lower widest widest widest
newest newest newest newest newest newest''')
