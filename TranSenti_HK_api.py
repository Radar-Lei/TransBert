# -*- coding: utf-8 -*-
from openai import OpenAI
import os
from pathlib import Path
import datetime
import pandas as pd
import time
import csv
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


def read_api_keys_from_csv(filepath):
    """
    Read API keys from a CSV file without headers and return as a list of strings.
    Args:
        filepath (str): Path to the CSV file containing API keys
    Returns:
        list: List of API keys as strings
    """
    keys = []
    try:
        with open(filepath, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row:  # Skip empty rows
                    keys.append(row[0])
        return keys
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def keyword_prefilter(directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Define keywords for filtering in different languages
    keywords = [
        #
        "train", "travel", "trip", "旅行", "出行", "でんしゃ", "りょこう", "トリップ",
        "기차", "여행", "트립", " voyage", "tren", "viaje", "excursión",

        # General transit terms
        "地铁", "地鐵", "metro", "subway", "métro", "地下鉄", "メトロ", "지하철", "전철", "지하철역",
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
        "Express de l'aéroport", "エアポートエクスプレス", "Expreso del Aeropuerto",

        # HK station name list
        '金鐘', '金钟', 'Admiralty', 'アドミラルティ', '애드미럴티', 'Amirauté', 'Almirantazgo', '機場', '机场', 'Airport', 'エアポート', '에어포트', 'Aéroport', 'Aeropuerto', '博覽館', '博览馆', 'AsiaWorld-Expo', 'アジアワールドエキスポ', '아시아월드엑스포', '柯士甸', 'Austin', 'オースティン', '오스틴', '銅鑼灣', '铜锣湾', 'Causeway Bay', 'コーズウェイ・ベイ', '코즈웨이 베이', '中環', '中环', 'Central', 'セントラル', '센트럴', '柴灣', '柴湾', 'Chai Wan', 'チャイワン', '차이완', '車公廟', '车公庙', 'Che Kung Temple', 'チェクンミャオ', '차공묘', 'Temple Che Kung', 'Templo de Che Kung', '長沙灣', '长沙湾', 'Cheung Sha Wan', 'チャンサーワン', '창사완', '彩虹', 'Choi Hung', 'チョイホン', '쵀이헝', '第一城', 'City One', 'シティワン', '시티원', 'Cité Un', 'Ciudad Uno', '鑽石山', '钻石山', 'Diamond Hill', 'ダイヤモンドヒル', '다이아몬드 힐', 'Diamond Hill', 'Colina del Diamante', '迪士尼', 'Disneyland Resort', 'ディズニーランドリゾート', '디즈니랜드 리조트', 'Complexe Disneyland', 'Complejo Disneyland', '尖東', '尖东', 'East Tsim Sha Tsui', 'イースト・チムシャツイ', '이스트 침사추이', 'Tsim Sha Tsui Est', 'Tsim Sha Tsui Este', '粉嶺', '粉岭', 'Fanling', 'ファンリン', '판링', '火炭', 'Fo Tan', 'フォータン', '포탄', '炮台山', 'Fortress Hill', 'フォートレスヒル', '포트리스 힐', 'Colline de la Forteresse', 'Colina de la Fortaleza', '坑口', 'Hang Hau', 'ハンハウ', '항하우', '杏花邨', 'Heng Fa Chuen', 'ヘンファチュン', '헝파촌', '恆安', '恒安', 'Heng On', 'ヘンオン', '헝온', '紅磡', 'Hung Hom', 'フンホム', '훙험', '佐敦', 'Jordan', 'ジョーダン', '조던', '錦上路', '锦上路', 'Kam Sheung Road', 'カムシンロード', '캄셍 로드', 'Route Kam Sheung', 'Carretera Kam Sheung', '九龍', '九龙', 'Kowloon', 'クーロン', '구룡', '九龍灣', '九龙湾', 'Kowloon Bay', 'クーロンベイ', '구룡만', 'Baie de Kowloon', 'Bahía de Kowloon', '九龍塘', '九龙塘', 'Kowloon Tong', 'クーロントン', '구룡통', '葵芳', 'Kwai Fong', 'クワイフォン', '퀘이팡', '葵興', '葵兴', 'Kwai Hing', 'クワイヒン', '퀘이힝', '觀塘', '观塘', 'Kwun Tong', 'クーントン', '관통', '荔枝角', 'Lai Chi Kok', 'ライチコク', '라이치콕', '荔景', 'Lai King', 'ライキン', '라이킹', '藍田', '蓝田', 'Lam Tin', 'ラムティン', '람틴', '羅湖', '罗湖', 'Lo Wu', 'ロウー', '로우', '康城', 'LOHAS Park', 'ロハスパーク', '로하스 파크', 'Parc LOHAS', 'Parque LOHAS', '樂富', '乐富', 'Lok Fu', 'ロックフー', '록푸', '落馬洲', '落马洲', 'Lok Ma Chau', 'ロックマチャウ', '록마차우', '朗屏', 'Long Ping', 'ロンピン', '롱핑', '馬鞍山', '马鞍山', 'Ma On Shan', 'マーオンシャン', '마안산', '美孚', 'Mei Foo', 'メイフー', '메이푸', '旺角', 'Mong Kok', 'モンコク', '몽콕', '旺角東', '旺角东', 'Mong Kok East', 'モンコクイースト', '몽콕 이스트', 'Mong Kok Est', 'Mong Kok Este', '南昌', 'Nam Cheong', 'ナムチョン', '남창', '牛頭角', '牛头角', 'Ngau Tau Kok', 'ンガウタウコク', '우타우콕', '北角', 'North Point', 'ノースポイント', '노스 포인트', '奧運', '奥运', 'Olympic', 'オリンピック', '올림픽', 'Olympique', 'Olímpico', '寶琳', '宝琳', 'Po Lam', 'ポーラム', '보람', '太子', 'Prince Edward', 'プリンスエドワード', '프린스 에드워드', 'Prince Edward', 'Príncipe Eduardo', '鰂魚涌', '鲗鱼涌', 'Quarry Bay', 'クォリーベイ', '쿼리 베이', '馬場', '马场', 'Racecourse', 'レースコース', '레이스코스', 'Hippodrome', 'Hipódromo', '西灣河', '西湾河', 'Sai Wan Ho', 'サイワンホー', '사이완호', '沙田', 'Sha Tin', 'シャーティン', '사틴', '沙田圍', '沙田围', 'Sha Tin Wai', 'シャーティンウェイ', '사틴웨이', '深水埗', 'Sham Shui Po', 'シャムシュイポー', '삼수이보', '筲箕灣', '筲箕湾', 'Shau Kei Wan', 'シャウケイワン', '사우케이완', '石硤尾', '石硖尾', 'Shek Kip Mei', 'シェキップメイ', '석합미', '石門', '石门', 'Shek Mun', 'シェクムン', '석문', '上水', 'Sheung Shui', 'シェンシュイ', '상수', '上環', '上环', 'Sheung Wan', 'シェンワン', '상환', '兆康', 'Siu Hong', 'シウホン', '시오홍', '欣澳', 'Sunny Bay', 'サニーベイ', '써니 베이', 'Baie ensoleillée', 'Bahía Soleada', '太古', 'Tai Koo', 'タイクー', '타이쿠', '大埔墟', 'Tai Po Market', 'タイポーマーケット', '타이포 마켓', 'Marché de Tai Po', 'Mercado de Tai Po', '大水坑', 'Tai Shui Hang', 'タイシュイハン', '타이수이항', '大圍', '大围', 'Tai Wai', 'タイワイ', '타이와이', '太和', 'Tai Wo', 'タイウォ', '타이워', '大窩口', '大窝口', 'Tai Wo Hau', 'タイウォハウ', '타이워하우', '天后', 'Tin Hau', 'ティンハウ', '틴하우', '天水圍', '天水围', 'Tin Shui Wai', 'ティンシュイワイ', '틴수이와이', '調景嶺', '调景岭', 'Tiu Keng Leng', 'ティウキンレン', '티우깡링', '將軍澳', '将军澳', 'ツェンジンオー', '청관오', 'Tseung Kwan', '尖沙咀', 'Tsim Sha Tsui', 'チムシャツイ', '침사추이', '青衣', 'Tsing Yi', 'チンイー', '칭이', '荃灣', '荃湾', 'Tsuen Wan', 'ツェンワン', '츄완', '荃灣西', '荃湾西', 'Tsuen Wan West', 'ツェンワンウェスト', '츄완 웨스트', 'Tsuen Wan Ouest', 'Tsuen Wan Oeste', '屯門', '屯门', 'Tuen Mun', 'トゥエンムン', 'тун문', '東涌', '东涌', 'Tung Chung', 'トゥンチョン', '퉁충', '大學', '大学', 'University', 'ユニバーシティ', '대학', 'Université', 'Universidad', '灣仔', '湾仔', 'ワンチャイ', '완차이', 'Wan Chai', '黃大仙', '黄大仙', 'Wong Tai Sin', 'ウォンタイシン', '웡타이신', '烏溪沙', '乌溪沙', 'ウーカイシャー', '우카이사', 'Wu Kai Sha', '油麻地', 'ヤウマーテイ', '야우마티', 'Yau Ma Tei', '油塘', 'Yau Tong', 'ヤウトン', '야우통', 'Yau Tong', '元朗', 'ユンロン', '윈롱', 'Yuen Long']
    
    total_posts = 0
    prefiltered_posts = 0
    
    for csv_file in Path(directory).glob('*.csv'):
        output_filename = f"prefiltered_{csv_file.name}"
        output_path = Path(output_directory) / output_filename
        
        if output_path.exists():
            print(f"File {output_filename} already exists. Skipping.")
            continue
        
        print(f"\nPrefiltering file: {csv_file} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            df = pd.read_csv(csv_file,index_col=0)
            df.reset_index(drop=False, inplace=True)
            df = df.rename(columns={df.columns[0]: "post_id"})
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        print(f"Total rows in {csv_file.name}: {len(df)}")
        total_posts += len(df)
        
        # Filter posts containing any of the keywords
        filtered_posts = []
        
        for index, row in df.iterrows():
            text = row['text']
                
            # Case-insensitive keyword matching
            text_lower = text.lower()
            if any(keyword.lower() in text_lower for keyword in keywords):
                filtered_posts.append(row)
        
        prefiltered_posts += len(filtered_posts)
        
        if filtered_posts:
            output_df = pd.DataFrame(filtered_posts)
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Saved {len(filtered_posts)} prefiltered posts out of {len(df)} posts in this file.")
    
    print(f"\nSaved \033[1;33m{prefiltered_posts}\033[0m prefiltered posts out of \033[1;33m{total_posts}\033[0m total posts.")
    return prefiltered_posts > 0  # Return True if any posts were prefiltered



def datafilter(directory, output_directory, api_keys=None):
    """
    Filter data using parallel processing with multiple API keys.
    Each CSV file is split into chunks and processed in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import math
    
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    if not api_keys:
        # Use default key if none provided
        api_keys = ["sk-7c87ef2add054e439095db9b18c921e9"]
    
    total_post_counter = 0
    valid_post_counter = 0
    
    def process_chunk(chunk_df, api_key):
        """Process a chunk of data using the specified API key"""
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        chunk_valid_posts = []
        
        for _, row in chunk_df.iterrows():
            each_post = row['text']
            if type(each_post) != str:
                continue
            if len(each_post) > 512:
                continue
                
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are a data filter"},
                        {"role": "user", "content": f"以下用户社交媒体发表的post是否是乘客对地铁公交服务质量、地铁公交环境相关的评价, 可能涉及到Reliability, Crowdedness, Comfort, Safety and securit, Waiting conditions, Service facilities等方面, 只回答'是'或'否', 不要回答你的分析内容。注意有些posts并非是真正的评价地铁服务地铁系统, 有可能只是提到了地铁, metro, subway等关键词 : {each_post}"},
                    ],
                    stream=False
                )
                
                print(f"\033[1;33m{each_post}\033[0m")
                
                response_text = response.choices[0].message.content.strip()
                if '\n</think>\n\n' in response_text:
                    response_text = response_text.split('\n</think>\n\n')[-1]
                
                if any(word in response_text for word in ['是', '对', 'yes', '确实']):
                    chunk_valid_posts.append(row.to_dict())
                print(f"is related to transit service: {response_text}")
                
            except Exception as e:
                print(f"Error processing with API key {api_key}: {e}")
                # If rate limited or other API error, sleep and try again
                if "rate" in str(e).lower() or "limit" in str(e).lower():
                    time.sleep(5)
                    try:
                        # Try one more time
                        response = client.chat.completions.create(
                            model="deepseek-reasoner",
                            messages=[
                                {"role": "system", "content": "You are a data filter"},
                                {"role": "user", "content": f"以下用户社交媒体发表的post是否是乘客对地铁公交服务质量、地铁公交环境相关的评价, 可能涉及到Reliability, Crowdedness, Comfort, Safety and securit, Waiting conditions, Service facilities, travel experience, travel time等方面, 只回答'是'或'否', 不要回答你的分析内容。注意有些posts并非是真正的评价地铁服务地铁系统, 有可能只是提到了地铁, metro, subway等关键词 : {each_post}"},
                            ],
                            stream=False
                        )
                        
                        response_text = response.choices[0].message.content.strip()
                        if '\n</think>\n\n' in response_text:
                            response_text = response_text.split('\n</think>\n\n')[-1]
                        
                        if any(word in response_text for word in ['是', '对', 'yes', '确实']):
                            chunk_valid_posts.append(row.to_dict())
                        print(f"Retry successful: {response_text}")
                    except:
                        print("Retry failed")
                        
        return chunk_valid_posts
    
    for csv_file in Path(directory).glob('*.csv'):
        output_filename = f"cleaned_{csv_file.name}"
        output_path = Path(output_directory) / output_filename
        
        if output_path.exists():
            print(f"File {output_filename} already exists. Skipping.")
            continue
        
        print(f"\nProcessing file: {csv_file} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            df = pd.read_csv(csv_file)
            print(f"Total rows in {csv_file.name}: {len(df)}")
            total_post_counter += len(df)
            
            # Split dataframe into chunks based on number of API keys available
            num_chunks = min(len(api_keys), max(1, len(df) // 20))  # At least 1 chunk, max 50 rows per API key
            chunk_size = math.ceil(len(df) / num_chunks)
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            
            print(f"Processing {len(df)} rows in {len(chunks)} chunks using {min(len(api_keys), len(chunks))} API keys")
            
            # Process chunks in parallel
            valid_posts = []
            with ThreadPoolExecutor(max_workers=min(len(api_keys), len(chunks))) as executor:
                future_to_chunk = {
                    executor.submit(process_chunk, chunk, api_keys[i % len(api_keys)]): i 
                    for i, chunk in enumerate(chunks)
                }
                
                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        chunk_valid_posts = future.result()
                        valid_posts.extend(chunk_valid_posts)
                        print(f"Chunk {chunk_index+1}/{len(chunks)} completed with {len(chunk_valid_posts)} valid posts")
                    except Exception as exc:
                        print(f"Chunk {chunk_index+1} generated an exception: {exc}")
            
            valid_post_counter += len(valid_posts)
            
            if valid_posts:
                output_df = pd.DataFrame(valid_posts)
                output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Saved {len(valid_posts)} valid posts to {output_filename}")
        
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
    
    print(f"\nSaved \033[1;33m{valid_post_counter}\033[0m filtered posts out of \033[1;33m{total_post_counter}\033[0m total posts.")
    return valid_post_counter > 0

if __name__ == "__main__":
    """
    prxosy set
    screen -ls
    run 'ollama serve' in "screen" instance of terminal first
    python TranSent_HK.py > data_filer_log.txt
    use absolute paths for multi-platform usage
    """

    api_ls = read_api_keys_from_csv("Data/DeepSeek_keys.csv")
    
    original_dir = "Data/Twitter_Hong Kong"
    prefiltered_dir = "prefiltered_results_HK"
    cleaned_dir = "cleaned_results_HK"
    sentiment_dir = "senti_results"

    start_time = datetime.datetime.now()
    print(f"Start time for data prefiltering: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # First apply keyword-based prefiltering
    # has_prefiltered_data = keyword_prefilter(original_dir, prefiltered_dir)
    prefilter_end_time = datetime.datetime.now()
    time_used = (prefilter_end_time - start_time).total_seconds() / 60
    print(f"Time used for data prefiltering: {time_used:.2f} minutes")
    
    # Only proceed with LLM filtering if prefiltering found any relevant posts
    print(f"Start time for LLM data filtering: {prefilter_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    datafilter(prefiltered_dir, cleaned_dir, api_keys=api_ls)
    end_time = datetime.datetime.now()

    # Release model and GPU memory
    # os.system("ollama stop deepseek-r1:32b")
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