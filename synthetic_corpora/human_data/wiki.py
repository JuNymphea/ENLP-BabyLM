from datasets import load_dataset, Dataset, load_from_disk, load_from_disk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import os, re, nltk

# dataset = load_dataset("wikipedia", language="en", date="20220301", cache_dir="/Users/shaoshao/Desktop/ENLP/homework/FinalProject/code/Constituency/")
# wiki = dataset["train"]

# dataset.save_to_disk("wikipedia_en_dataset")
# nltk.download("punkt")
dataset = load_from_disk("wikipedia_en_dataset")
wiki = dataset["train"]

def is_valid_english_sentence(sent):
    return re.fullmatch(r"[A-Za-z0-9\s.,!?'\-\"():;\[\]]+", sent) is not None

output_dir = "wiki_cleaned_output"
os.makedirs(output_dir, exist_ok=True)

sentences_per_file = 100_000_000
buffer_size = 1000

file_index = 0
file_sentence_count = 0
total_sentences = 0
buffer = []

def get_output_path(index):
    return os.path.join(output_dir, f"cleaned_wiki_sentences_part_{index}.txt")

output_path = get_output_path(file_index)

with open(output_path, "w", encoding="utf-8") as f_out:
    for article in tqdm(wiki, desc="Processing articles"):
        text = article["text"]
        sentences = sent_tokenize(text)

        for sent in sentences:
            sent = sent.strip()

            if len(sent) < 10:
                continue
            if re.match(r"^[.,!?;:'\"()\[\]-]", sent):
                continue
            if not re.match(r".+[.!?]$", sent):
                continue
            if not is_valid_english_sentence(sent):
                continue

            buffer.append(sent + "\n")
            total_sentences += 1
            file_sentence_count += 1

            if len(buffer) >= buffer_size:
                f_out.writelines(buffer)
                buffer = []

            if file_sentence_count >= sentences_per_file:
                if buffer:
                    f_out.writelines(buffer)
                    buffer = []

                f_out.close()
                file_index += 1
                file_sentence_count = 0
                output_path = get_output_path(file_index)
                f_out = open(output_path, "w", encoding="utf-8")

    if buffer:
        f_out.writelines(buffer)
    f_out.close()


