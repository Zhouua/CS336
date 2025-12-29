from webbrowser import get
import regex as re

# 支持直接运行和模块导入两种方式
try:
    from . import pretokenization_example
except ImportError:
    import pretokenization_example

# 输入：input_path: 输入文件路径 vocab_size: 最大的词汇表大小 special_tokens: 特殊token
# 输出：vocab: token_id到token_bytes的字典 merges: BPE中merge的tokens
def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    freq_table = {}
    # 打开文件
    with open(input_path, "rb") as f:
        # 使用提供的函数找到分块边界并进行多线程处理
        num_processes = 4
        boundaries = pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]): # 配对成每个区间(start, end)
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # 不对special_tokens分词
            chunk_freq = pretokenization(chunk, special_tokens)
            # 合并频率表
            for word, count in chunk_freq.items():
                freq_table[word] = freq_table.get(word, 0) + count
    # 对vocab初始化。要把special_tokens放入低位，并用encode转换为bytes
    vocab = {i: special_tokens[i].encode("utf-8") for i in range(len(special_tokens))}
    vocab.update({i + len(special_tokens): bytes([i]) for i in range(256)})
    # 初始化merges
    merges = []
    # 进行merge
    merge(freq_table, vocab_size, vocab, merges, special_tokens)
    # print(merges)
    return (vocab, merges)
    

import regex as re
# 使用正则表达式对每个buffer进行预分词，不对special_tokens分词
def pretokenization(buffer: str, special_tokens: list[str] = None) -> dict[bytes, int]:
    dic = {}
    origin_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        # 先用special_pattern进行分隔文本，避免出现" <endoftext>"这样有前导空格的问题
        parts = re.split(f"({special_pattern})", buffer)
        # 每个part都进行预分词
        for part in parts:
            if not part: 
                continue
            if part in special_tokens:
                # 直接加入
                dic[part.encode("utf-8")] = dic.get(part.encode("utf-8"), 0) + 1
            else:
                matches = re.finditer(origin_pattern, part)
                for match in matches:
                    word = match.group().encode("utf-8")
                    dic[word] = dic.get(word, 0) + 1
    else:
        matches = re.finditer(origin_pattern, buffer)
        for match in matches:
            word = match.group().encode("utf-8")
            dic[word] = dic.get(word, 0) + 1
    # print(dic)
    return dic


# 对freq_table进行BPE
def merge(freq_table: dict[bytes, int], vocab_size: int, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
    # 将special_tokens转换为bytes集合，用于快速查找
    special_tokens_bytes = set(token.encode("utf-8") for token in special_tokens)
    # 将每个word转换为list[bytes]，方便后续替换
    word_splits = {}
    for word, freq in freq_table.items():
        # 如果是special token，跳过（不进行BPE合并）
        if word in special_tokens_bytes:
            continue
        word_splits[word] = [bytes([b]) for b in word] # 每个word对应list[bytes]
    
    # 计算所有pair的出现次数，计算一次，后续不再计算
    counts = {}
    for word, word_list in word_splits.items():
        freq = freq_table[word]
        for i in range(len(word_list) - 1):
            pair = (word_list[i], word_list[i + 1])
            counts[pair] = counts.get(pair, 0) + freq
    
    while len(vocab) < vocab_size:
        if not counts:
            break
        
        # 找到出现次数最多的pair
        best_pair = None
        best_count = 0
        for pair, count in counts.items():
            # 选择字典序更大的pair
            if best_pair is None or count > best_count or (count == best_count and pair > best_pair):
                best_pair = pair
                best_count = count
        # 忽略出现次数为0的pair
        if best_pair is None or best_count == 0:
            break
        
        # 将best_pair加入vocab和merges
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)
        
        # 更新word_splits: 在每个word中用new_token替换best_pair
        new_word_splits = {}
        for word, word_list in word_splits.items():
            new_list = []
            i = 0
            while i < len(word_list):
                # 如果当前位置和下一个位置构成best_pair，则合并
                if i < len(word_list) - 1 and word_list[i] == best_pair[0] and word_list[i + 1] == best_pair[1]:
                    # 更新count
                    # 原先(word_list[i - 1], word_list[i]) -> (word_list[i - 1], new_token)
                    # 和(word_list[i + 1], word_list[i + 2]) -> (new_token, word_list[i + 2])
                    freq = freq_table[word]
                    if new_list:
                        # counts[(word_list[i - 1], word_list[i])] -= freq
                        # counts[(word_list[i - 1], new_token)] = counts.get((word_list[i - 1], new_token), 0) + freq
                        # 要使用new_list[-1]而不是word_list[i - 1]
                        counts[(new_list[-1], word_list[i])] = counts.get((new_list[-1], word_list[i]), 0) - freq
                        counts[(new_list[-1], new_token)] = counts.get((new_list[-1], new_token), 0) + freq
                    if i + 2 < len(word_list):
                        counts[(word_list[i + 1], word_list[i + 2])] -= freq
                        counts[(new_token, word_list[i + 2])] = counts.get((new_token, word_list[i + 2]), 0) + freq
                    # 对best_pair的计数置为0，这里不能直接删除，因为后续其他word可能还包含best_pair
                    counts[(word_list[i], word_list[i + 1])] = 0
                    # 添加new_token
                    new_list.append(new_token)
                    i += 2
                else:
                    new_list.append(word_list[i])
                    i += 1
            new_word_splits[word] = new_list
        word_splits = new_word_splits
        
                

# vocab, merges = train_bpe("test.txt", 300, ["<endoftext>"])
# print(vocab)
# print(merges)


