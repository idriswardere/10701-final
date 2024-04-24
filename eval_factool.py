import json
import os
from dotenv import load_dotenv
from factool import Factool

INPUTS_PATH = "fever_inputs.jsonl"
LABELS_PATH = "fever_labels.jsonl"

def main():
    # Initialize a Factool instance with the specified keys. foundation_model could be either "gpt-3.5-turbo" or "gpt-4"
    factool_instance = Factool("gpt-4")

    with open(INPUTS_PATH, "r") as json_file:
        json_list = list(json_file)
        inputs = []
        for entry in json_list:
            inputs.append(json.loads(entry))

    with open(LABELS_PATH, "r") as json_file:
        json_list = list(json_file)
        labels = []
        for label in json_list:
            labels.append(json.loads(label))

    response_list = factool_instance.run(inputs[:15])

    pred_labels = []
    for output in response_list["detailed_information"]:
        pred = output["response_level_factuality"]
        if pred:
            pred_labels.append("SUPPORTS")
        else:
            pred_labels.append("NOT ENOUGH INFO")

    correct = 0
    total = 0
    for label, pred_label in zip(labels, pred_labels):
        if label == "SUPPORTS" and pred_label == "SUPPORTS":
            correct += 1
        elif label == ("NOT ENOUGH INFO" or label == "REFUTES") and pred_label != "SUPPORTS":
            correct += 1
        total += 1

    print(f"====\nAccuracy: {correct/total}\n====")


if __name__ == "__main__":
    load_dotenv()
    main()