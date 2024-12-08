import torch
from ollama import chat
from ollama import ChatResponse
import os
from pathlib import Path
import numpy as np
"""
First manually download the model, the merge them into one .gguf file using:
cat qwen2.5-32b-instruct-q4_k_m*.gguf > qwen2.5-32b-instruct-q4_k_m.gguf

pip install gguf
pip3 install sentencepiece
pip install numpy==1.26.3
"""
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be")

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
except ImportError:
    raise ImportError (
        "This example requires classes from the 'transformers' Python package. " 
        "You can install it with 'pip install transformers'"
    )


def sentiment_evaluation(inputs):
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

    # Load the model using the Transformer classes
    print (f"\n > Loading model '{model_name}'from HuggingFace...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # hf_model = AutoModel.from_pretrained(model_name, gguf_file=filename)
    # hf_tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=filename)

    # Bring the model into llware.  These models were not trained on instruction following, 
    # so we set instruction_following to False
    inputs = hf_tokenizer(inputs, return_tensors="pt")
    with torch.no_grad():
        logits = hf_model(**inputs).logits
    
    logits = logits.numpy()
    logits = logits[0]
    # Normalize using sigmoid
    normalized = 1 / (1 + np.exp(-logits))
    # Find index of max value for coloring
    max_idx = np.argmax(normalized)

    # Create formatted string with colored max value
    result = "Negative:{}, Neutral:{}, Positive:{}".format(
        f"\033[1;33m{normalized[0]:.4f}\033[0m" if max_idx == 0 else f"{normalized[0]:.4f}",
        f"\033[1;33m{normalized[1]:.4f}\033[0m" if max_idx == 1 else f"{normalized[1]:.4f}",
        f"\033[1;33m{normalized[2]:.4f}\033[0m" if max_idx == 2 else f"{normalized[2]:.4f}"
    )

    print(result)
    return normalized

def datafilter(directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    total_post_counter = 0
    valid_post_counter = 0
    valid_posts = []
    original_columns = None  # Store original column names
    
    for csv_file in Path(directory).glob('*.csv'):
        print(f"\nProcessing file: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Total rows in {csv_file.name}: {len(df)}")
        total_post_counter += len(df)
        # Store column names from first file
        if original_columns is None:
            original_columns = df.columns
        
        for index, row in df.iterrows():
            each_post = row['微博正文']
            print(f"当前微博是否与公交(地铁)服务相关: {each_post}")
            response: ChatResponse = chat(model='qwen2.5:32b', messages=[
                {
                    'role': 'user',
                    'content': f"以下内容是否与地铁服务质量、地铁环境相关, 只回答'是'或'否', 不要回答你的分析内容 : {each_post}",
                },
            ])
            print(f"\033[1;33m{response.message.content}\033[0m")
            
            try:
                response_text = response.message.content.strip().lower()
                if response_text in ['是', '对', 'yes', '确实']:
                    valid_post_counter += 1
                    valid_posts.append(row.to_dict())  # Convert row to dictionary
                    print(f"有效微博数: {valid_post_counter}")
            except AttributeError:
                print("Warning: Invalid response format")
            except Exception as e:
                print(f"Error processing response: {e}")
            print("\n")
    
    if valid_posts:
        output_df = pd.DataFrame(valid_posts, columns=original_columns)  # Create DataFrame with original columns
        output_path = Path(output_directory) / 'filtered_posts.csv'
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved \033[1;33m{valid_post_counter}\033[0m filtered posts out of \033[1;33m{total_post_counter}\033[0m total posts to {output_path}")

print("Starting script...")

if __name__ == "__main__":
    # use absolute paths for multi-platform usage
    directory = "/home/TransBert/Data/Shenzhen"
    output_directory = "TransBert/cleaned_data"
    datafilter(directory, output_directory)