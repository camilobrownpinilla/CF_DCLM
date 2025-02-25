import argparse
import os
import json
import yaml
import datasets
import tarfile
import gzip
import uuid
import subprocess

from transformers import AutoTokenizer
from tqdm import tqdm

        
class DSDataLoader:
    def __init__(self, yaml, task_config):
        self.yaml = yaml
        self.task_config = task_config
        self.tokenizer = AutoTokenizer.from_pretrained(**yaml['tokenizer'])
        self.out_path = yaml['output_dir']
        self.task_label = task_config['label']

        self.tokenizer.truncation_side = 'left'
        # NOTE: gpt-neox-20b-pii-special does not have a pad token natively.
        # Therefore, I set it to a special token (which are all eos tokens)
        # to avoid loss computation on pads.
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.answer_dict = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'E': 4
        }

    def _get_answer_from_choices(self, choices, answer, choice_key=""):
        if 'arc' in self.task_label or 'openbook_qa' in self.task_label: # Allenai format
            if isinstance(answer, str):
                idx = choices["label"].index(answer)
                return choices["text"][idx]
            elif isinstance(answer, int):
                return choices['text'][answer-1] # adjust 1 indexing

        else:  # Generic format
            if isinstance(answer, str) and answer in self.answer_dict:
                    if choice_key:
                        answer = choices[choice_key][self.answer_dict[answer]]
                    else:
                        answer = choices[self.answer_dict[answer]]
            else:
                try:
                    answer = choices[int(answer)]
                except Exception as e:
                    print(f'Error: {e}, {self.task_label} cannot be handled. Try modifying the script.') 
            
            return answer


    def _format_qa_pair(self, context, question, answer, question_prelimiter="", continuation_delimiter=" "):
        return f"{context}{question_prelimiter}{question}{continuation_delimiter}{answer}"

    def process(self, data):
        total_tokens = 0

        question_prelimiter = self.task_config.get("question_prelimiter", "")
        continuation_delimiter = self.task_config.get("continuation_delimiter", " ")

        context_col = self.task_config.get("context_col", "")
        question_col = self.task_config.get("question_col", "question")
        choice_col = self.task_config.get("choice_col", "")
        choice_key = self.task_config.get("choice_index", "")
        answer_col = self.task_config.get("answer_col", "answer")

        # Open tar file once at the beginning
        tar_path = os.path.join(self.out_path, f'{self.task_label}.tar')
        with tarfile.open(tar_path, 'w') as tar:
            for c, entry in tqdm(enumerate(data), desc='Formatting', colour='yellow', unit='example', total=len(data)):
                context = entry.get(context_col, "")
                question = entry.get(question_col, "")
                choices = entry.get(choice_col)
                
                # Handle CoQA multiple QA pairs
                if isinstance(question, list):
                    answers = entry.get(answer_col, {"input_text": []})
                    qa_pairs = zip(question, answers["input_text"])

                # Handle SQuAD format:
                elif self.task_label == 'squad':
                    answer = entry.get(answer_col)['text']
                    if answer: answer = answer[0] # Answers formatted like ['answer']; some are missing
                    else:
                        with open('./bad_squad_examples.txt', 'a') as f:
                            f.write(f"Iter: {c} \tQuestion: {question}, \tAnswer{answer}, \tAnswer Col: {entry.get(answer_col)}\n")
                    qa_pairs = [(question, answer)]

                else:
                    answer = entry.get(answer_col)
                    if choices:
                        answer = self._get_answer_from_choices(choices, answer, choice_key)
                    qa_pairs = [(question, answer)]

                # Process all QA pairs for the current entry
                for q, a in qa_pairs:
                    formatted_entry = self._format_qa_pair(
                        context, q, a, question_prelimiter, continuation_delimiter
                    )
                    tokens = self.tokenizer(formatted_entry, 
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.tokenizer.model_max_length)['input_ids']
                    example_name = f'example-{c}_{uuid.uuid4()}'
                    json_path = os.path.join(self.out_path, f'{self.task_label}_{example_name}.json.gz')
                    with gzip.open(json_path, 'wt') as json_file:
                        json.dump(tokens, json_file)
                    tar.add(json_path, arcname=os.path.basename(json_path))
                    os.remove(json_path)
                    total_tokens += len(tokens)

        # Each example should be exactly model_max_length tokens long
        expected_tokens = (c + 1) * self.tokenizer.model_max_length 
        if self.task_label != 'coqa':
        # Necessary if statement b/c structure of coqa invalidates formula
            assert total_tokens == expected_tokens, f"\033[31mERROR:\033[0m Did not process expected number of tokens ({total_tokens / expected_tokens})"
        print(f'Successfully processed {total_tokens} tokens, saved to {tar_path}')

        return


class DatasetProcessor:
    def __init__(self, cfg):
        self.yaml_config = cfg
        self.output_dir = self.yaml_config['output_dir']
    def process_datasets(self):
        icl_tasks = self.yaml_config.get("icl_tasks", [])
        os.makedirs(self.output_dir, exist_ok=True)

        for task in icl_tasks:
            label = task["label"]
            dataset_uri = task["dataset_uri"]
            dataset_second = task.get("dataset_second", None)

            print(f"\033[33mProcessing dataset:\033[0m {label}")

            if dataset_uri == "cais/mmlu":
                dataset = datasets.load_dataset(dataset_uri, dataset_second, split="auxiliary_train")
            elif dataset_second:
                dataset = datasets.load_dataset(dataset_uri, dataset_second, split="train")
            else:
                dataset = datasets.load_dataset(dataset_uri, split="train")

            loader = DSDataLoader(self.yaml_config, task)
            loader.process(dataset)


def read_yaml(file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
        
def prepare_format_yaml(_cfg):
    """Prepare the the yaml used for `load_ds` to be used for `format`

    Args:
        cfg: Loaded yaml config
    """
    # Avoid overwriting original cfg
    cfg = _cfg 
    cfg['tar_paths'] = f"{cfg['output_dir']}/*.tar"

    tmp_yaml_path = f'{os.getcwd()}/tmp-format-yaml-{uuid.uuid4()}.yaml'
    with open(tmp_yaml_path, 'w') as f:
        yaml.dump(cfg, f)

    return tmp_yaml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to yaml config')
    args = parser.parse_args()
    cfg = read_yaml(args.config)

    processor = DatasetProcessor(cfg)
    processor.process_datasets()

    format_yaml = prepare_format_yaml(cfg)
    format_script_path = f'{os.path.dirname(os.path.realpath(__file__))}/format.py'

    subprocess.run(['python', format_script_path, format_yaml])
    os.remove(format_yaml)
