import json
import os
from dotenv import load_dotenv
from factool import Factool

INPUTS_PATH = "fever_inputs.jsonl"
LABELS_PATH = "fever_labels.jsonl"
RESPONSE_PATH = 'factool_response.json'

def create_response():
    factool_instance = Factool("gpt-4")

    with open(INPUTS_PATH, "r") as json_file:
        json_list = list(json_file)
        inputs = []
        for entry in json_list:
            inputs.append(json.loads(entry))

    response_list = factool_instance.run(inputs)

    with open(RESPONSE_PATH, 'w') as file:
        json.dump(response_list, file)


def eval():
    with open(RESPONSE_PATH, 'r') as file:
        response_list = json.load(file)

    with open(LABELS_PATH, "r") as json_file:
        json_list = list(json_file)
        labels = []
        for label in json_list:
            labels.append(json.loads(label))

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
    # create_response()
    eval()