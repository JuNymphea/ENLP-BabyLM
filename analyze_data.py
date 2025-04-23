import json
import math
import argparse
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Analyze bracketed structure statistics from a sequence file.")
parser.add_argument("--input", "-i", required=True, help="Input file path (sequence format)")
parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
args = parser.parse_args()

input_file = args.input
output_file = args.output

depth_counter = defaultdict(int)
leaf_counter = defaultdict(int)
width_counter = defaultdict(int)
branch_avg_counter = defaultdict(int)
branch_var_counter = defaultdict(int)
branch_std_counter = defaultdict(int)
symmetry_counter = defaultdict(int)

def get_nesting_depth(seq_line):
    nums = seq_line.strip().split()
    stack = []
    max_depth = 0
    for num in nums:
        if stack and stack[-1] == num:
            stack.pop()
        else:
            stack.append(num)
            max_depth = max(max_depth, len(stack))
    return max_depth

def count_leaf_nodes(seq_line):
    tokens = seq_line.strip().split()
    stack = []
    leaf_count = 0
    for i, token in enumerate(tokens):
        if not token.isdigit():
            continue
        if stack and stack[-1][0] == token:
            token_id, open_idx = stack.pop() 
            if i - open_idx == 1:
                leaf_count += 1
        else:
            stack.append((token, i))
    return leaf_count


def compute_width(seq_line):
    tokens = seq_line.strip().split()
    stack = []
    max_width = 0
    for token in tokens:
        if not token.isdigit():
            continue
        if stack and stack[-1] == token:
            stack.pop()
        else:
            stack.append(token)
            max_width = max(max_width, len(stack))
    return max_width

def compute_branching_factors(seq_line):
    tokens = seq_line.strip().split()
    stack = []
    children = {}
    for i, token in enumerate(tokens):
        if not token.isdigit():
            continue
        if stack and stack[-1][0] == token:
            stack.pop()
        else:
            stack.append((token, i))
            if len(stack) >= 2:
                parent_id, _ = stack[-2]
                children.setdefault(parent_id, 0)
                children[parent_id] += 1
    if not children:
        return 0.0, 0.0, 0.0
    counts = list(children.values())
    avg = sum(counts) / len(counts)
    var = sum((x - avg) ** 2 for x in counts) / len(counts)
    std = math.sqrt(var)
    return avg, var, std

def compute_symmetry_score(seq_line):
    tokens = seq_line.strip().split()
    stack = []
    spans = []
    for i, token in enumerate(tokens):
        if not token.isdigit():
            continue
        if stack and stack[-1][0] == token:
            open_token, start = stack.pop()
            spans.append((start, i))
        else:
            stack.append((token, i))
    if not spans:
        return 0.0
    symmetric_count = 0
    total_count = 0
    for start, end in spans:
        if end - start <= 2:
            continue
        sub = tokens[start + 1:end]
        if sub == sub[::-1]:
            symmetric_count += 1
        total_count += 1
    if total_count == 0:
        return 0.0
    return round(symmetric_count / total_count, 4)

def remove_crossing_tags(tag_sequence_str):
    tags = tag_sequence_str.split()
    
    from collections import defaultdict
    tag_positions = defaultdict(list)

    for idx, tag in enumerate(tags):
        tag_positions[tag].append(idx)

    tag_intervals = defaultdict(list)
    for tag, positions in tag_positions.items():
        if len(positions) % 2 != 0:
            # print(f"⚠️ Tag {tag} appears {len(positions)} times, not an even number. Ignoring.")
            continue
        for i in range(0, len(positions), 2):
            start, end = sorted((positions[i], positions[i+1]))
            tag_intervals[tag].append((start, end))

    non_projective_tags = set()
    all_pairs = []
    for tag, intervals in tag_intervals.items():
        for start, end in intervals:
            all_pairs.append((start, end, tag))

    for i in range(len(all_pairs)):
        s1, e1, tag1 = all_pairs[i]
        for j in range(i+1, len(all_pairs)):
            s2, e2, tag2 = all_pairs[j]
            if (s1 < s2 < e1 < e2) or (s2 < s1 < e2 < e1):
                non_projective_tags.add(tag1)
                non_projective_tags.add(tag2)

    cleaned = [tag for tag in tags if tag not in non_projective_tags]
    return ' '.join(cleaned), len(non_projective_tags)


with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

crossing = 0

for line in tqdm(lines, desc="Processing lines"):
    if not line.strip():
        continue

    data = remove_crossing_tags(line)

    line = data[0]

    if data[1] != '':
        crossing += int(data[1])

    depth = get_nesting_depth(line)
    leaf = count_leaf_nodes(line)
    width = compute_width(line)
    avg, var, std = compute_branching_factors(line)
    symmetry = compute_symmetry_score(line)

    depth_counter[depth] += 1
    leaf_counter[leaf] += 1
    width_counter[width] += 1

    branch_avg_counter[round(avg, 2)] += 1
    branch_var_counter[round(var, 2)] += 1
    branch_std_counter[round(std, 2)] += 1

    symmetry_counter[round(symmetry, 2)] += 1

output = {
    "depth": dict(sorted(depth_counter.items())),
    "leaves": dict(sorted(leaf_counter.items())),
    "width": dict(sorted(width_counter.items())),
    "branching": {
        "avg": dict(sorted(branch_avg_counter.items())),
        "var": dict(sorted(branch_var_counter.items())),
        "std": dict(sorted(branch_std_counter.items()))
    },
    "symmetry": dict(sorted(symmetry_counter.items())),
    "crossing": crossing
}

with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(output, f_out, indent=2)

