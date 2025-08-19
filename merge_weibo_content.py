#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 senti_results_SZ 文件夹中所有 CSV 文件的微博正文到一个 txt 文件
"""

import os
import pandas as pd
import glob
from pathlib import Path

def merge_weibo_content(input_folder, output_file):
    """
    从指定文件夹中的所有 CSV 文件提取微博正文，合并到一个 txt 文件
    
    Args:
        input_folder (str): 包含 CSV 文件的文件夹路径
        output_file (str): 输出的 txt 文件路径
    """
    
    # 获取文件夹中所有 CSV 文件
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"在文件夹 {input_folder} 中没有找到 CSV 文件")
        return
    
    print(f"找到 {len(csv_files)} 个 CSV 文件")
    
    all_content = []
    processed_count = 0
    
    for csv_file in csv_files:
        try:
            print(f"正在处理文件: {os.path.basename(csv_file)}")
            
            # 读取 CSV 文件
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # 检查是否存在"微博正文"列
            if '微博正文' in df.columns:
                # 提取微博正文，去除空值
                content = df['微博正文'].dropna().astype(str)
                
                # 过滤掉空字符串和只包含空白字符的内容
                content = content[content.str.strip() != '']
                
                all_content.extend(content.tolist())
                processed_count += 1
                print(f"  - 提取了 {len(content)} 条微博正文")
            else:
                print(f"  - 警告：文件 {csv_file} 中没有找到'微博正文'列")
                
        except Exception as e:
            print(f"  - 错误：处理文件 {csv_file} 时出错: {str(e)}")
    
    # 写入合并后的内容到 txt 文件
    if all_content:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for content in all_content:
                    f.write(content.strip() + '\n')
            
            print(f"\n合并完成！")
            print(f"成功处理了 {processed_count} 个 CSV 文件")
            print(f"总共提取了 {len(all_content)} 条微博正文")
            print(f"结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"写入文件时出错: {str(e)}")
    else:
        print("没有提取到任何微博正文内容")

if __name__ == "__main__":
    # 设置输入和输出路径
    input_folder = "/Users/leida/TransBert/senti_results_SZ"
    output_file = "/Users/leida/TransBert/merged_weibo_content.txt"
    
    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹不存在: {input_folder}")
        exit(1)
    
    # 执行合并
    merge_weibo_content(input_folder, output_file)
