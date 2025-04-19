import stanza, random, os
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Process
from stanza.pipeline.core import DownloadMethod

GPUS = [0, 1, 2, 3]
INPUT_FILE = "../wiki_cleaned_output/dependency.txt"
OUTPUT_BASE = "../wiki_encoded_output/dependency_split"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def count_crossing_arcs(sentence):
    count = 0
    for i, wi in enumerate(sentence.words):
        for j, wj in enumerate(sentence.words):
            if wi.head == 0 or wj.head == 0 or i == j:
                continue
            ai, bi = sorted([i, wi.head - 1])
            aj, bj = sorted([j, wj.head - 1])
            if (ai < aj < bi < bj) or (aj < ai < bj < bi):
                count += 1
    return count // 2

def replace_pairs_with_random(s: str):
    nums = list(map(int, s.strip().split()))
    positions = defaultdict(list)
    for idx, num in enumerate(nums):
        positions[num].append(idx)
    for num, indices in positions.items():
        if len(indices) == 2:
            rand_num = random.randint(0, 500)
            nums[indices[0]] = rand_num
            nums[indices[1]] = rand_num
        else:
            raise ValueError(f"Error: Number {num} doesn't occur twice.")
    return ' '.join(map(str, nums)), len(nums)

def compress_dependency_structure(text, nlp):
    doc = nlp(text)
    result = ""
    for sentence in doc.sentences:
        projective = count_crossing_arcs(sentence)
        words = sentence.words
        n = len(words)
        before_tags = [[] for _ in range(n)]
        after_tags = [[] for _ in range(n)]
        tag_to_span = {}

        for word in words:
            if word.head == 0:
                continue
            dep_id = word.id
            head_id = word.head
            dep_idx = dep_id - 1
            head_idx = head_id - 1
            tag = str(dep_id)
            span = abs(dep_idx - head_idx)
            tag_to_span[tag] = span

            if dep_idx < head_idx:
                after_tags[dep_idx].append(tag)
                before_tags[head_idx].append(tag)
            else:
                after_tags[head_idx].append(tag)
                before_tags[dep_idx].append(tag)

        for i in range(n):
            before_tags[i].sort(key=lambda tag: tag_to_span[tag])
            after_tags[i].sort(key=lambda tag: tag_to_span[tag], reverse=True)

        pieces = []
        for i in range(n):
            pieces.append(' '.join(before_tags[i]))
            pieces.append(' '.join(after_tags[i]))

        result = " ".join(pieces)

    return result, projective

def process_lines(lines, gpu_id, part_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Starting part {part_idx}")
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,depparse, lemma', use_gpu=True, download_method=DownloadMethod.REUSE_RESOURCES)

    output_path = os.path.join(OUTPUT_BASE, f"part_{part_idx}.txt")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines, desc=f"[GPU {gpu_id}] part {part_idx}"):
            sentence = line.strip()
            if not sentence:
                f_out.write("\n")
                continue
            try:
                compressed, _ = compress_dependency_structure(sentence, nlp)
                encoded, _ = replace_pairs_with_random(compressed)
                f_out.write(encoded + "\n")
            except Exception as e:
                print(f"[GPU {gpu_id}] Error: {e}")
                f_out.write("\n")

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    num_gpus = len(GPUS)
    chunk_size = len(all_lines) // num_gpus + 1

    processes = []
    for i, gpu_id in enumerate(GPUS):
        chunk = all_lines[i*chunk_size : (i+1)*chunk_size]
        p = Process(target=process_lines, args=(chunk, gpu_id, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
