import os
import stanza
import re
import random
from tqdm import tqdm
from multiprocessing import Process
from stanza.pipeline.core import DownloadMethod

GPUS = [0, 1, 2, 3]
INPUT_FILE = "../wiki_cleaned_output/constituency.txt"
OUTPUT_BASE = "../wiki_encoded_output/constituency_split"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def is_valid_english_sentence(sent):
    return re.fullmatch(r"[A-Za-z0-9\s.,!?'\-\"():;\[\]]+", sent) is not None

def encode_bracket_structure(parse_line):
    stack = []
    encoded = []
    local_bracket_count = 0
    for char in parse_line:
        if char == '(':
            rand_num = random.randint(0, 500)
            stack.append(rand_num)
            encoded.append(str(rand_num))
            local_bracket_count += 1
        elif char == ')':
            if stack:
                match = stack.pop()
                encoded.append(str(match))
                local_bracket_count += 1
    return " ".join(encoded), local_bracket_count

def process_lines(lines, gpu_id, part_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Starting part {part_idx}")

    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', use_gpu=True, download_method=DownloadMethod.REUSE_RESOURCES)

    output_path = os.path.join(OUTPUT_BASE, f"part_{part_idx}.txt")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for sentence in tqdm(lines, desc=f"[GPU {gpu_id}] part {part_idx}"):
            sentence = sentence.strip()
            if (
                len(sentence) < 10 or
                re.match(r"^[.,!?;:'\"()\[\]-]", sentence) or
                not re.match(r".+[.!?]$", sentence) or
                not is_valid_english_sentence(sentence)
            ):
                f_out.write("\n")
                continue
            try:
                doc = nlp(sentence)
                for sent in doc.sentences:
                    tree = str(sent.constituency)
                    encoded, _ = encode_bracket_structure(tree)
                    f_out.write(encoded + "\n")
            except Exception as e:
                print(f"[GPU {gpu_id}] Error parsing: {sentence}")
                print(e)
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
