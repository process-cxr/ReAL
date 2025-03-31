import os
import re
import json
import tqdm
import logging
import argparse
from vllm import LLM, SamplingParams
import ast

def setup_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def _parse_response(response: str, candidate_answer: str, question: str, logger: logging.Logger) -> int:
    patterns = [
        r".*['\"]?(yes|no)\.?['\"]?[.!]?$",
        r".*I can answer\s+['\"]?(yes|no)['\"]?[.!]?",
        r".*I would say\s+['\"]?(yes|no)['\"]?[.!]?",
        r".*I must say\s+['\"]?(yes|no)['\"]?[.!]?",
        (r".*my (final )?judgment is\s+['\"]?(yes|no)['\"]?[.!]?", 2),
        r".*I would judge the candidate answer as\s+['\"]?(yes|no)['\"]?[.!]?",
        r".*\s+['\"]?(yes|no)['\"]?,? the candidate( answer)? is",
        r".*[jJ]udgment:\s+['\"]?(yes|no)\.?['\"]?",
    ]
    correct_patterns = [r"candidate( answer)? is correct", r"candidate's correct"]

    acceptable = ""
    if response.lower().startswith("yes"):
        acceptable = "Yes"
    elif response.lower().startswith("no"):
        acceptable = "No"
    else:
        for pattern in patterns:
            match_idx = 1
            if isinstance(pattern, (list, tuple)):
                pattern, match_idx = pattern

            matched = re.match(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)

            if matched:
                acceptable = matched.group(match_idx).capitalize()
                break
        
        if not acceptable:
            for pattern in correct_patterns:
                matched = re.search(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if matched:
                    acceptable = "Yes"
                    break

        if not acceptable:
            logger.warning(f"Invalid response to `{question}` & `{candidate_answer}`: {response}")

    return int(acceptable == "Yes")

def load_data(answer_txt, prediction_txt, prompt_txt):
    with open(answer_txt, 'r') as f:
        answer_data = f.readlines()

    with open(prediction_txt, 'r') as f:
        prediction_data = f.readlines()

    with open(prompt_txt, 'r') as f:
        prompt_template = f.read()

    return answer_data, prediction_data, prompt_template

def evaluate_model(llm_base, answer_data, prediction_data, prompt_template, output_file, json_file, logger: logging.Logger):
    record = []
    count = 0

    with open(output_file, 'w') as fw:
        for i in tqdm.trange(len(answer_data)):
            query, answers = answer_data[i].split('\t')
            prediction = prediction_data[i].split('\t')[1]
            query = query.strip()
            prediction = prediction.strip()
            # answers = ", ".join(eval(answers.strip('"').strip("'"))) 有报错，太多特殊引号问题
            try:
                answers = ", ".join(ast.literal_eval(answers))
            except:
                answers = str(answers)
            
            instruction = prompt_template.format(q=query, answers=answers, candidate_answer=prediction)

            sampling_params = SamplingParams(temperature=0.0, max_tokens=100, top_p=1.0)
            output = llm_base.generate(instruction, sampling_params)

            response = output[0].outputs[0].text.replace('\n', ' ')
            logger.info(f"QuestionId: {i+1} --- Query: {query} --- Answer: {answers} --- Response: {response}")

            judgement = _parse_response(response, prediction, query, logger)
            count += judgement

            item = [str(i), response]
            fw.write('\t'.join(item) + '\n')

            temp_record = {
                'query': query,
                'answers': answers,
                'prediction': prediction,
                'judgement': judgement,
                'response': response
            }
            record.append(temp_record)

    with open(json_file, "w") as jf:
        json.dump(record, jf, indent=4, ensure_ascii=False)

    accuracy = count / len(answer_data)
    return accuracy, count

def main(args):
    os.makedirs(args.base_out_path, exist_ok=True)
    
    logger = setup_logger(os.path.join(args.base_out_path, 'evaluation.log'))

    model_path = ''
    llm_base = LLM(model_path, tensor_parallel_size=4)

    output_file = os.path.join(args.base_out_path, 'response.txt')
    json_file = os.path.join(args.base_out_path, 'response_record.txt')
    
    answer_data, prediction_data, prompt_template = load_data(args.answer_txt, args.prediction_txt, args.prompt_txt)

    accuracy, count = evaluate_model(llm_base, answer_data, prediction_data, prompt_template, output_file, json_file, logger)
    
    logger.info(f"Accuracy: {accuracy}, Count: {count}, Prediction file: {args.prediction_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument('--base_out_path', type=str, required=True, help="Base output path for logs and results")
    parser.add_argument('--answer_txt', type=str, required=True, help="Path to the file containing the answers")
    parser.add_argument('--prediction_txt', type=str, required=True, help="Path to the file containing the predictions")
    parser.add_argument('--prompt_txt', type=str, required=True, help="Path to the file containing the prompt template")

    args = parser.parse_args()
    main(args)


