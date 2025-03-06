import torch
from openai import OpenAI
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


def keyword_prefilter(directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Define keywords for filtering in different languages
    keywords = [
        # General transit terms
        "地铁", "地鐵", "metro", "subway", "métro", "地下鉄", "メトロ",
        "港铁", "港鐵", "MTR", "轨道交通", "軌道交通", "rail transit", 
        "transport ferroviaire", "transporte ferroviario",
        
        # Hong Kong specific lines
        "港岛线", "港島線", "Island Line", "Ligne de l'Île", "アイランドライン", "Línea de la Isla",
        "荃湾线", "荃灣線", "Tsuen Wan Line", "Ligne Tsuen Wan", "ツェンワンライン", "Línea Tsuen Wan",
        "观塘线", "觀塘線", "Kwun Tong Line", "Ligne Kwun Tong", "クーンフォンライン", "Línea Kwun Tong",
        "南港岛线", "南港島線", "South Island Line", "Ligne Sud de l'Île", "サウスアイランドライン", 
        "Línea Sur de la Isla", "東涌线", "東涌線", "Tung Chung Line", "Ligne Tung Chung", 
        "トゥンチョンライン", "Línea Tung Chung", "迪士尼线", "迪士尼線", "Disneyland Resort Line", 
        "Ligne Disneyland", "ディズニーリゾートライン", "Línea de la Reserva de Disney",
        "東鐵線", "东铁线", "East Rail Line", "Ligne Est", "イーストレールライン", 
        "Línea Este del Ferrocarril", "屯馬線", "屯马线", "Tuen Ma Line", "Ligne Tuen Ma", 
        "トゥエンマーライン", "Línea Tuen Ma", "機場快線", "机场快线", "Airport Express", 
        "Express de l'aéroport", "エアポートエクスプレス", "Expreso del Aeropuerto"
    ]
    
    total_posts = 0
    prefiltered_posts = 0
    
    for csv_file in Path(directory).glob('*.csv'):
        output_filename = f"prefiltered_{csv_file.name}"
        output_path = Path(output_directory) / output_filename
        
        if output_path.exists():
            print(f"File {output_filename} already exists. Skipping.")
            continue
        
        print(f"\nPrefiltering file: {csv_file} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        df = pd.read_csv(csv_file)
        print(f"Total rows in {csv_file.name}: {len(df)}")
        total_posts += len(df)
        
        # Filter posts containing any of the keywords
        filtered_posts = []
        
        for index, row in df.iterrows():
            text = row['text']
            if type(text) != str:
                continue
                
            # Skip posts that are too long
            if len(text) > 512:
                continue
                
            # Case-insensitive keyword matching
            text_lower = text.lower()
            if any(keyword.lower() in text_lower for keyword in keywords):
                filtered_posts.append(row.to_dict())
        
        prefiltered_posts += len(filtered_posts)
        
        if filtered_posts:
            output_df = pd.DataFrame(filtered_posts)
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Saved {len(filtered_posts)} prefiltered posts out of {len(df)} posts in this file.")
    
    print(f"\nSaved \033[1;33m{prefiltered_posts}\033[0m prefiltered posts out of \033[1;33m{total_posts}\033[0m total posts.")
    return prefiltered_posts > 0  # Return True if any posts were prefiltered








def datafilter(directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    total_post_counter = 0
    valid_post_counter = 0
    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    for csv_file in Path(directory).glob('*.csv'):
        output_filename = f"cleaned_{csv_file.name}"
        output_path = Path(output_directory) / output_filename

        if output_path.exists():
            print(f"File {output_filename} already exists. Skipping.")
            continue

        print(f"\nProcessing file: {csv_file} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        df = pd.read_csv(csv_file)
        print(f"Total rows in {csv_file.name}: {len(df)}")
        total_post_counter += len(df)

        valid_posts = []

        for index, row in df.iterrows():
            each_post = row['text']
            if type(each_post) != str:
                continue
            if len(each_post) > 512:
                continue

            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a data filter"},
                    {"role": "user", "content": f"以下用户社交媒体发表的post是否是乘客对地铁公交服务质量、地铁公交环境相关的评价, 可能涉及到Reliability, Crowdedness, Comfort, Safety and securit, Waiting conditions, Service facilities等方面, 只回答'是'或'否', 不要回答你的分析内容。注意有些posts并非是真正的评价地铁服务地铁系统, 有可能只是提到了地铁, metro, subway等关键词 : {each_post}"},
                ],
                stream=False
            )

            print(f"\033[1;33m{each_post}\033[0m")

            try:
                response_text = response.choices[0].message.content.strip().lower()
                if '\n</think>\n\n' in response_text:
                    response_text = response_text.split('\n</think>\n\n')[-1]
                
                # Now check if the final response indicates a valid subway service review
                if any(word in response_text for word in ['是', '对', 'yes', '确实']):
                    valid_post_counter += 1
                    valid_posts.append(row.to_dict())  # Convert row to dictionary
                print(f"is related to transit service: {response_text}")

            except AttributeError:
                print("Warning: Invalid response format")
            except Exception as e:
                print(f"Error processing response: {e}")

        if valid_posts:
            output_df = pd.DataFrame(valid_posts)  # Create DataFrame with original columns
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nSaved \033[1;33m{valid_post_counter}\033[0m filtered posts out of \033[1;33m{total_post_counter}\033[0m total posts.")

if __name__ == "__main__":
    """
    prxosy set
    screen -ls
    run 'ollama serve' in "screen" instance of terminal first
    python TranSent_HK.py > data_filer_log.txt
    """
    # use absolute paths for multi-platform usage
    original_dir = "/home/TransBert/Data/Twitter_Hong Kong"
    prefiltered_dir = "/home/TransBert/prefiltered_results_HK"
    cleaned_dir = "/home/TransBert/cleaned_results_HK"
    sentiment_dir = "/home/TransBert/senti_results"

    # original_dir = "./Data/Twitter_Hong Kong"
    # prefiltered_dir = "./prefiltered_results_HK"
    # cleaned_dir = "./cleaned_results_HK"
    # sentiment_dir = "./senti_results"

    start_time = datetime.datetime.now()
    print(f"Start time for data prefiltering: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # First apply keyword-based prefiltering
    has_prefiltered_data = keyword_prefilter(original_dir, prefiltered_dir)
    prefilter_end_time = datetime.datetime.now()
    time_used = (prefilter_end_time - start_time).total_seconds() / 60
    print(f"Time used for data prefiltering: {time_used:.2f} minutes")
    
    # Only proceed with LLM filtering if prefiltering found any relevant posts
    print(f"Start time for LLM data filtering: {prefilter_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    datafilter(prefiltered_dir, cleaned_dir)
    end_time = datetime.datetime.now()

    # Release model and GPU memory
    os.system("ollama stop deepseek-r1:32b")
    print(f"End time for data cleaning: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_used = (end_time - prefilter_end_time).total_seconds() / 60
    print(f"Time used for LLM data filtering: {time_used:.2f} minutes")
    total_time = (end_time - start_time).total_seconds() / 60
    print(f"Total processing time: {total_time:.2f} minutes")

    
    # print(f"Start time for sentiment analysis: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # sentiment_evaluation(cleaned_dir, sentiment_dir)
    # end_time = datetime.datetime.now()
    # print(f"End time for sentiment analysis: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # time_used = (end_time - start_time).total_seconds() / 60
    # print(f"Time used for sentiment analysis: {time_used:.2f} minutes")