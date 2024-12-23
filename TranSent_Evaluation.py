import torch
from ollama import chat
from ollama import ChatResponse
import os
from pathlib import Path
import numpy as np
import datetime
import pandas as pd
"""
First manually download the model, the merge them into one .gguf file using:
cat qwen2.5-32b-instruct-q4_k_m*.gguf > qwen2.5-32b-instruct-q4_k_m.gguf

pip install gguf
pip3 install sentencepiece
pip install numpy==1.26.3

pip proxy: proxy set
"""


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be")

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
except ImportError:
    raise ImportError (
        "This example requires classes from the 'transformers' Python package. " 
        "You can install it with 'pip install transformers'"
    )


def sentiment_evaluation(directory, output_directory):
    if torch.cuda.is_available():
        device = "cuda:1"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Device: {device}")
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Load the model using the Transformer classes
    print (f"\n > Loading model '{model_name}' ")
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_model.to(device)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for csv_file in Path(directory).glob('*.csv'):
        print(f"\nProcessing file: {csv_file} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        df = pd.read_csv(csv_file)
        print(f"Total rows in {csv_file.name}: {len(df)}")

        store_posts = []

        for index, row in df.iterrows():
            each_post = row['微博正文'] if '微博正文' in row else row['Blog']
            inputs = hf_tokenizer(each_post, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = hf_model(**inputs).logits
            
            logits = logits.cpu().numpy()
            logits = logits[0]
            # Normalize using sigmoid
            normalized = 1 / (1 + np.exp(-logits))
            # if 
            if not (normalized[0] == max(normalized) or normalized[2] == max(normalized) and abs(max(normalized) - normalized[1]) > 0.1):
                continue
            # row to dict such that we do not need to specify columns in the end
            post_dict = row.to_dict()
                # Add normalized sentiment scores
            post_dict['Negative'] = normalized[0]
            post_dict['Neutral'] = normalized[1]
            post_dict['Positive'] = normalized[2]
            store_posts.append(post_dict)  # Convert row to dictionary

        output_df = pd.DataFrame(store_posts)
        output_filename = f"senti_{csv_file.name}"
        output_path = os.path.join(output_directory, output_filename)
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')


def datafilter(directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    total_post_counter = 0
    valid_post_counter = 0

    for file_path in Path(directory).glob('*.[cx]*'):
        if file_path.suffix.lower() in ['.csv', '.xlsx']:
            print(f"\nProcessing file: {file_path} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            print(f"Total rows in {file_path.name}: {len(df)}")
            total_post_counter += len(df)

        valid_posts = []

        for index, row in df.iterrows():
            each_post = row['微博正文'] if '微博正文' in row else row['Blog']
            if len(each_post) > 512:
                continue
            response: ChatResponse = chat(model='qwen2.5:32b', messages=[
                {
                    'role': 'user',
                    'content': f"以下内容是否与地铁服务质量、地铁环境相关, 只回答'是'或'否', 不要回答你的分析内容 : {each_post}",
                },
            ])
            
            try:
                response_text = response.message.content.strip().lower()
                if response_text in ['是', '对', 'yes', '确实']:
                    valid_post_counter += 1
                    valid_posts.append(row.to_dict())  # Convert row to dictionary
            except AttributeError:
                print("Warning: Invalid response format")
            except Exception as e:
                print(f"Error processing response: {e}")

        if valid_posts:
            output_df = pd.DataFrame(valid_posts)  # Create DataFrame with original columns
            output_filename = f"cleaned_{file_path.name}"
            output_path = Path(output_directory) / output_filename
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nSaved \033[1;33m{valid_post_counter}\033[0m filtered posts out of \033[1;33m{total_post_counter}\033[0m total posts.")

if __name__ == "__main__":
    """
    screen -ls
    run 'ollama serve' in "screen" instance of terminal first
    python TranSent.py > data_filer_log.txt
    """
    # use absolute paths for multi-platform usage

    original_dir = os.path.join(os.getcwd(), "TransBert", "data_eva")
    cleaned_dir = os.path.join(os.getcwd(), "TransBert", "cleaned_eva")
    sentiment_dir = os.path.join(os.getcwd(), "TransBert", "senti_results_eva")

    start_time = datetime.datetime.now()
    print(f"Start time for data cleaning: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    datafilter(original_dir, cleaned_dir)
    end_time = datetime.datetime.now()

    # to realease model and GPU memory
    os.system("ollama stop qwen2.5:32b")
    print(f"End time for data cleaning: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_used = (end_time - start_time).total_seconds() / 60
    print(f"Time used for data cleaning: {time_used:.2f} minutes")
    print("\n")
    
    print(f"Start time for sentiment analysis: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # sentiment_evaluation(cleaned_dir, sentiment_dir)
    end_time = datetime.datetime.now()
    print(f"End time for sentiment analysis: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_used = (end_time - start_time).total_seconds() / 60
    print(f"Time used for sentiment analysis: {time_used:.2f} minutes")