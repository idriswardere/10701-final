import json
import random

DATA_PATH = f"./data/paper_dev.jsonl"
NUM_INPUTS = 100

def create_factool_inputs():
    with open(DATA_PATH, "r") as json_file:
        json_list = list(json_file)
        claims, labels = [], []
        idxs = list(range(7405))
        random.Random(0).shuffle(idxs)
        for i in idxs[:NUM_INPUTS]:
            entry = json_list[i]
            entry_j = json.loads(entry)
            claims.append(entry_j["claim"])
            labels.append(entry_j["label"])

    new_inputs_jsonl = []
    new_labels_jsonl = []
    for i in range(len(claims)):
        new_entry = {}
        new_entry["prompt"] = ""
        new_entry["category"] = "kbqa"
        new_entry["response"] = claims[i]
        new_inputs_jsonl.append(new_entry)
        new_labels_jsonl.append(labels[i])

    with open('fever_inputs.jsonl', 'w') as outfile:
        for entry in new_inputs_jsonl:
            json.dump(entry, outfile)
            outfile.write('\n')

    with open('fever_labels.jsonl', 'w') as outfile:
        for entry in new_labels_jsonl:
            json.dump(entry, outfile)
            outfile.write('\n')     

if __name__ == "__main__":
    create_factool_inputs()
