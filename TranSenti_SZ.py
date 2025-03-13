# -*- coding: utf-8 -*-
from ollama import chat
from ollama import ChatResponse
import os
from pathlib import Path
import datetime
import pandas as pd
import glob
import shutil
import argparse
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

def datafilter(directory, output_directory, batch_size=20, model_name='deepseek-r1:32b'):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    total_post_counter = 0
    valid_post_counter = 0

    for csv_file in Path(directory).glob('*.csv'):
        # Create temporary directories for chunks
        tmp_input_dir = Path(directory) / "tmp_input" / csv_file.stem
        tmp_output_dir = Path(output_directory) / "tmp_output" / csv_file.stem
        tmp_input_dir.mkdir(parents=True, exist_ok=True)
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"cleaned_{csv_file.name}"
        output_path = Path(output_directory) / output_filename

        if output_path.exists():
            print(f"File {output_filename} already exists. Skipping.")
            # Clean up temporary directories if output file exists
            shutil.rmtree(tmp_input_dir, ignore_errors=True)
            shutil.rmtree(tmp_output_dir, ignore_errors=True)
            continue

        print(f"\nProcessing file: {csv_file} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            df['post_id'] = df.index
            print(f"Total rows in {csv_file.name}: {len(df)}")
            total_post_counter += len(df)

            # Split the DataFrame into chunks and save to temporary files
            chunk_size = 100  # Adjust chunk size as needed
            for i, chunk in enumerate(np.array_split(df, len(df) // chunk_size + 1)):
                chunk.to_csv(tmp_input_dir / f"chunk_{i}.csv", index=False, encoding='utf-8-sig')

            # Process each chunk
            chunk_files = sorted(tmp_input_dir.glob('chunk_*.csv'), key=lambda x: int(x.stem.split('_')[1]))
            for chunk_file in chunk_files:
                print(f"Processing chunk: {chunk_file.name}")
                chunk_df = pd.read_csv(chunk_file)

                # Check if the processed chunk file already exists
                processed_chunk_path = tmp_output_dir / f"processed_{chunk_file.name}"
                if processed_chunk_path.exists():
                    print(f"Processed chunk file {processed_chunk_path.name} already exists. Skipping.")
                    continue

                # Filter out posts that are not strings or too long
                valid_chunk_df = chunk_df[chunk_df['微博正文'].apply(lambda x: isinstance(x, str) and len(x) <= 512)].copy()

                valid_posts = []
                # Process in batches
                for i in range(0, len(valid_chunk_df), batch_size):
                    batch_df = valid_chunk_df.iloc[i:i+batch_size]
                    
                    # Format batch posts with original post_ids
                    formatted_posts = "\n\n".join([f"Post {row['post_id']}: {row['微博正文']}" for _, row in batch_df.iterrows()])

                    print(f"Processing sub-batch {i//batch_size + 1}/{(len(valid_chunk_df) + batch_size - 1)//batch_size} of chunk {chunk_file.name}")

                    try:
                        response: ChatResponse = chat(model=model_name, messages=[
                            {
                                'role': 'user',
                                'content': f"""请判断以下{len(batch_df)}个社交媒体post是否是对地铁服务、地铁运营相关的评价。
                                可能涉及到地铁的Reliability, Crowdedness, Comfort, Safety and security, Waiting conditions, Service facilities, travel experience等方面。
                                请注意, 有些posts并非真正评价地铁服务或地铁系统, 可能只是提到了地铁、metro、subway等关键词, 我们需要真正评价地铁服务或地铁系统的posts, 租房等广告信息不能算是与地铁服务相关。

                                {formatted_posts}

                                请以JSON格式回答,每个post对应一个以上述判断标准的是或否的结论, 使用原始post ID:
                                {{
                                "post[ID]": "是/否",
                                "post[ID]": "是/否",
                                ...
                                }}
                                仅返回JSON格式,结果一定要完整{len(batch_df)}条。""",
                            },
                        ])

                        response_text = response.message.content.strip()
                        print(f"Response text: {response_text}")
                        # Extract JSON part if there's explanatory text
                        if '{' in response_text and '}' in response_text:
                            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                            results = eval(json_str)

                            # Process results
                            for j, (_, row) in enumerate(batch_df.iterrows()):
                                id_str = str(int(row['post_id']))
                                if any(id_str in key and results[key] == "是" for key in results):
                                    valid_post_counter += 1
                                    # Append the original row from the chunk_df
                                    valid_posts.append(chunk_df.loc[row.name].to_dict())
                        else:
                            print(f"Invalid response format")

                    except Exception as e:
                        print(f"Error processing batch response: {e}")
                        # Optionally, save the failed chunk for later inspection
                        chunk_file.rename(tmp_input_dir / f"failed_{chunk_file.name}")
                        continue # Continue to the next chunk even if one fails

                if valid_posts:
                    output_chunk_df = pd.DataFrame(valid_posts)
                    output_chunk_df.to_csv(tmp_output_dir / f"processed_{chunk_file.name}", index=False, encoding='utf-8-sig')

            # Merge processed chunks
            processed_chunk_files = sorted(tmp_output_dir.glob('processed_chunk_*.csv'), key=lambda x: int(x.stem.split('_')[1].split('.')[0]))
            if processed_chunk_files:
                merged_df = pd.concat([pd.read_csv(f) for f in processed_chunk_files])
                merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            else:
                print(f"No valid posts found in {csv_file.name}")

        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
        finally:
            pass

    print(f"\nSaved \033[1;33m{valid_post_counter}\033[0m filtered posts out of \033[1;33m{total_post_counter}\033[0m total posts.")

import numpy as np  # Import numpy

if __name__ == "__main__":
    """
    prxosy set
    screen -ls
    "OLLAMA_SCHED_SPREAD=1 ollama serve"
    conda activate rag
    python TranSenti_SZ.py >> data_filer_log_SZ.txt
    use absolute paths for multi-platform usage
    """
    parser = argparse.ArgumentParser(description="Data filtering script for TranSenti project.")
    parser.add_argument("--model", default="deepseek-r1:32b", help="The Ollama model to use for sentiment analysis. Alternative models: qwq:32b")
    args = parser.parse_args()

    original_dir = "Data/Shenzhen"
    cleaned_dir = "cleaned_results_SZ"

    start_time = datetime.datetime.now()
    print(f"Start time for data processing: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Directly process data without prefiltering
    datafilter(original_dir, cleaned_dir, model_name=args.model)
    end_time = datetime.datetime.now()

    # Release model and GPU memory
    # os.system(f"ollama stop {args.model}") # Don't stop a model the user may be using
    print(f"End time for data cleaning: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_used = (end_time - start_time).total_seconds() / 60
    print(f"Time used for data processing: {time_used:.2f} minutes")
