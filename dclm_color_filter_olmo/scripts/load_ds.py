import os
import json
import yaml
import datasets


class ARCDataLoader:
    def __init__(self, config, task_config):
        self.config = config
        self.task_config = task_config
        self.answer_dict = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'E': 4
        }
        self.max_tokens = 2048
        self.pad_token = "<|endoftext|>"

    def process(self, data):
        output = []
        current_example = ""
        num_tokens = 0
        cur_tokens = 0
        continuation_delimiter = self.task_config.get("continuation_delimiter", " ")

        question_col = self.task_config.get("question_col", "question")
        choice_col = self.task_config.get("choice_col", "")
        answer_col = self.task_config.get("answer_col", "answer")

        for c, entry in enumerate(data):
            question = entry.get(question_col, "")
            choices = entry.get(choice_col)
            answer_letter = entry.get(answer_col)

            idx = (choices["label"]).index(answer_letter)
            answer = choices["text"][idx]

            formatted_entry = f"{question}{continuation_delimiter}{answer}"
            current_example += formatted_entry
            cur_tokens += len(formatted_entry)

            if current_example:
                    padding_needed = self.max_tokens - (num_tokens % self.max_tokens)
                    if padding_needed < self.max_tokens:
                        current_example += self.pad_token * padding_needed
                    
                    output.append(current_example)
                    num_tokens += len(current_example)

                    current_example = ""
                    cur_tokens = 0

            if c % 1000 == 0:
                print(f"Processed {c} entries in ARC dataset.")

        if current_example:
            padding_needed = self.max_tokens - (num_tokens % self.max_tokens)
            if padding_needed < self.max_tokens:
                current_example += self.pad_token * padding_needed
            output.append(current_example)
            num_tokens += len(current_example)

        return output, num_tokens


class CoQADataLoader:
    def __init__(self, config, task_config):
        self.config = config
        self.task_config = task_config
        self.max_tokens = 2048
        self.pad_token = "<|endoftext|>"

    def process(self, data):
        output = []
        current_example = ""
        num_tokens = 0
        cur_tokens = 0

        question_prelimiter = self.task_config.get("question_prelimiter", "")
        continuation_delimiter = self.task_config.get("continuation_delimiter", " ")

        context_col = self.task_config.get("context_col", "")
        question_col = self.task_config.get("question_col", "question")
        answer_col = self.task_config.get("answer_col", "answer")

        for c, entry in enumerate(data):
            context = entry.get(context_col, "")
            questions = entry.get(question_col, [])
            answers = entry.get(answer_col, {"input_text": []})

            for i in range(len(questions)):
                question = questions[i]
                answer = answers["input_text"][i]

                formatted_entry = f"{context}{question_prelimiter}{question}{continuation_delimiter}{answer}"
                current_example += formatted_entry
                cur_tokens += len(formatted_entry)

                if current_example:
                    padding_needed = self.max_tokens - (num_tokens % self.max_tokens)
                    if padding_needed < self.max_tokens:
                        current_example += self.pad_token * padding_needed
                    
                    output.append(current_example)
                    num_tokens += len(current_example)

                    current_example = ""
                    cur_tokens = 0

            if c % 10000 == 0:
                print(f"Processed {c} entries in CoQA dataset.")

        if current_example:
            padding_needed = self.max_tokens - (num_tokens % self.max_tokens)
            if padding_needed < self.max_tokens:
                current_example += self.pad_token * padding_needed
            output.append(current_example)
            num_tokens += len(current_example)

        return output, num_tokens


class GenericDataLoader:
    def __init__(self, config, task_config):
        self.config = config
        self.task_config = task_config
        self.answer_dict = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'E': 4
        }
        self.max_tokens = 2048
        self.pad_token = "<|endoftext|>"

    def process(self, data):
        output = []
        current_example = ""
        num_tokens = 0
        cur_tokens = 0

        question_prelimiter = self.task_config.get("question_prelimiter", "")
        continuation_delimiter = self.task_config.get("continuation_delimiter", " ")

        context_col = self.task_config.get("context_col", "")
        question_col = self.task_config.get("question_col", "question")
        choice_col = self.task_config.get("choice_col", "")
        choice_key = self.task_config.get("choice_index", "")
        answer_col = self.task_config.get("answer_col", "answer")

        for c, entry in enumerate(data):
            context = entry.get(context_col, "")
            question = entry.get(question_col, "")
            choices = entry.get(choice_col)
            answer = entry.get(answer_col)

            if choices:
                if isinstance(answer, str) and answer in self.answer_dict:
                    answer = choices[choice_key][self.answer_dict[answer]]
                else:
                    answer = choices[int(answer)]
            else:
                try:
                    answer = int(answer)
                except Exception as e:
                    answer = answer
            
            formatted_entry = f"{context}{question_prelimiter}{question}{continuation_delimiter}{answer}"
            current_example += formatted_entry
            cur_tokens += len(formatted_entry)

            if current_example:
                padding_needed = self.max_tokens - (num_tokens % self.max_tokens)
                if padding_needed < self.max_tokens:
                    current_example += self.pad_token * padding_needed
                
                output.append(current_example)
                num_tokens += len(current_example)

                current_example = ""
                cur_tokens = 0

            if c % 10000 == 0:
                print(f"Processed {c} entries in general dataset.")

        if current_example:
            padding_needed = self.max_tokens - (num_tokens % self.max_tokens)
            if padding_needed < self.max_tokens:
                current_example += self.pad_token * padding_needed
            output.append(current_example)
            num_tokens += len(current_example)

        return output, num_tokens


class DatasetProcessor:
    def __init__(self, yaml_path, output_dir):
        self.yaml_path = yaml_path
        self.output_dir = output_dir

    def load_yaml_config(self):
        print("Loading YAML configuration.")
        with open(self.yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def process_datasets(self):
        yaml_config = self.load_yaml_config()
        icl_tasks = yaml_config.get("icl_tasks", [])
        os.makedirs(self.output_dir, exist_ok=True)

        for task in icl_tasks:
            label = task["label"]
            dataset_uri = task["dataset_uri"]
            dataset_second = task.get("dataset_second", None)

            output_file = os.path.join(self.output_dir, f"{label}.jsonl")
            print(f"Processing dataset: {label}")

            if dataset_uri == "cais/mmlu":
                dataset = datasets.load_dataset(dataset_uri, dataset_second, split="auxiliary_train")
            elif dataset_second:
                dataset = datasets.load_dataset(dataset_uri, dataset_second, split="train")
            else:
                dataset = datasets.load_dataset(dataset_uri, split="train")

            if "allenai" in dataset_uri:
                loader = ARCDataLoader(yaml_config, task)
            elif "stanfordnlp/coqa" == dataset_uri:
                loader = CoQADataLoader(yaml_config, task)
            else:
                loader = GenericDataLoader(yaml_config, task)

            formatted_data, num_toks = loader.process(dataset)
            output_data = {"text": " ".join(formatted_data)}

            with open(output_file, 'w') as file:
                json.dump(output_data, file)
            
            with open('log_mmlu_padded.txt', 'a') as log:
                log.write(dataset_uri + ': ')
                log.write(str(num_toks) + ' tokens\n')

            print(f"Processed {num_toks} tokens")
            print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    yaml_path = "/n/netscratch/sham_lab/Everyone/cbrownpinilla/CF_DCLM/dclm_color_filter_olmo/configs/dclm/format_core-tasks-v3.yaml"
    output_dir = "/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/raw/old_core-task-trainsets-v3"

    processor = DatasetProcessor(yaml_path, output_dir)
    processor.process_datasets()