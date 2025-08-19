#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深圳地铁服务维度情感分析脚本
分析merged_weibo_content.txt中的微博内容，为6个服务维度分别生成positive和negative情感得分
"""

import re
import csv
import os
from collections import defaultdict

# 定义站点名称（从station_sentiment_rank.csv中提取）
STATION_NAMES = [
    "大拐弯", "通新岭", "景田", "深圳东", "塘坑", "民治", "福田", "坂田", "深圳西", "深湾",
    "上芬", "元芬", "阳台山东", "购物公园", "石厦", "平湖", "老街", "木棉湾", "布吉", "深圳北站",
    "长岭陂", "塘朗", "龙城广场", "双龙", "南联", "爱联", "大运", "荷坳", "永湖", "横岗",
    "六约", "丹竹头", "大芬", "草埔", "清湖", "龙华", "龙胜", "上塘", "红山", "白石龙",
    "民乐", "桥头", "塘尾", "马安山", "沙井", "车公庙", "深圳北", "黄贝岭", "清湖北", "竹村",
    "茜坑", "长湖", "观澜", "观澜湖", "松元厦", "牛湖", "田贝", "吉祥", "后海", "南山",
    "前海湾", "宝安", "碧海湾", "机场", "福永", "后亭", "松岗", "碧头", "会展中心", "太安",
    "盐田路", "盐田港西", "莲塘", "大新", "高新园", "竹子林", "华侨城", "科学馆", "新安", "香蜜湖",
    "宝体", "华强路", "国贸", "鲤鱼门", "深康", "深圳坪山", "侨城北", "留仙洞", "侨香", "世界之窗",
    "红树湾", "科苑", "登良", "海月", "湾厦", "东角头", "水湾", "海上世界", "赤湾", "蛇口港",
    "香梅北", "市民中心", "岗厦北", "华强北", "燕南", "湖贝", "香蜜", "坪洲", "西乡", "固戍",
    "白石洲", "深大", "桃园", "宝安中心", "侨城东", "岗厦", "罗湖", "桂湾", "前湾", "前湾公园",
    "妈湾", "铁路公园", "荔湾", "高新南", "粤海门", "深大南", "南山书城", "南油", "南油西", "荔林",
    "怡海", "梦海", "深圳机场", "沙井西", "福海西", "深圳机场北", "桃源村", "深圳", "光明城", "文体公园",
    "冬瓜岭", "平湖南", "木古", "上李朗", "双拥街", "禾花", "华南城", "体育中心", "八卦岭", "甘坑",
    "雪象", "岗头", "华为", "贝尔路", "坂田北", "五和", "光雅园", "南坑", "雅宝", "孖岭",
    "莲花村", "福田口岸", "福民", "沙头角", "海山", "深外高中", "莲塘口岸", "仙湖路", "梧桐山南", "福保",
    "西丽", "笋岗", "凤凰城", "通新岭‎", "莲花西", "大剧院", "新秀", "溪头", "松岗公园", "薯田埔",
    "合水口", "公明广场", "红花山", "楼村", "科学公园", "光明", "光明大街", "长圳", "上屋", "官田",
    "梅林关", "翰岭", "银湖", "翻身", "临海", "茶光", "龙井", "上沙", "皇岗村", "皇岗口岸",
    "红树湾南", "深圳湾公园", "下沙", "香梅", "梅景", "下梅林", "梅村", "上梅林", "泥岗", "红岭北",
    "园岭", "红岭", "红岭南", "鹿丹村", "人民南", "向西村", "文锦", "凉帽山", "洪湖‎", "华新‎",
    "黄木岗", "翠竹", "晒布", "莲花北", "少年宫‎", "水贝", "益田", "怡景", "布心", "百鸽笼",
    "长龙", "下水径", "上水径", "兴东", "洪浪北", "灵芝", "宝华", "珠光", "杨美", "华强南",
    "赤尾", "沙尾", "卫星厅", "3号航站楼", "国展南", "机场北", "会展城", "国展北", "国展", "沙田",
    "坑梓", "坪山中心", "坪山广场", "坪山围", "锦龙", "宝龙", "南约", "嶂背", "坳背", "四联",
    "六约北", "石芽岭", "罗湖北", "海上田园东", "左炮台东", "太子湾", "南光", "四海", "花果山", "南头古城",
    "中山公园", "同乐南", "上川", "流塘", "宝安客运站", "宝田一路", "平峦山", "西乡桃源", "钟屋南", "新安公园",
    "黄田", "兴围", "福围", "怀德", "桥头西", "海上田园南", "深理工", "中大", "圳美", "机场东",
    "自然博物馆西", "未来城", "燕子岭", "中芯国际", "综合保税区", "文化聚落", "站前路东", "坪山高铁站", "田心", "龙城公园",
    "大运中心", "黄阁坑", "愉园", "回龙埔", "尚景", "盛平", "龙园", "新塘围", "龙东", "宝龙同乐",
    "坪山", "新和", "六和", "坪环", "东纵纪念馆", "沙壆", "燕子湖", "石井", "技术大学", "比亚迪北",
    "龙背", "上梅林站", "盐田", "礼宾楼", "东区", "1#食堂", "总装", "生态", "创新", "龙坪",
    "六角", "蛇口西", "安托山", "西丽湖", "深云", "超级总部", "鸿安围", "盐田墟", "大梅沙", "小梅沙",
    "农林", "后瑞", "将围", "新庄", "上村", "下村", "李松蓢", "人才公园", "深圳湾口岸", "高新中",
    "北大", "深大丽湖", "福星", "步涌", "沙井古墟", "沙蚝", "蚝乡", "富坪", "白石塘", "低碳城",
    "坪西", "新生", "梨园", "坪地六联", "前海", "朗下", "月亮路", "民俗村", "茵特拉根", "大学城",
    "站前路东站"
]

# 定义服务维度及其关键词
SERVICE_DIMENSIONS = {
    'Reliability': {
        'positive': ['准时', '可靠', '准点', '正常运行', '稳定', '按时', '不延误'],
        'negative': ['延误', '晚点', '故障', '停运', '不准时', '取消', '中断', '停止', '断电', '维修']
    },
    'Crowdedness': {
        'positive': ['空旷', '不拥挤', '有座位', '人少', '宽敞', '舒适', '空座', '排队有序'],
        'negative': ['拥挤', '挤', '人多', '爆满', '人山人海', '站不稳', '挤不上', '排队', '满员', '没座位']
    },
    'Staff': {
        'positive': ['服务好', '态度好', '热情', '礼貌', '帮助', '友善', '专业', '微笑', '耐心', '周到'],
        'negative': ['态度差', '服务差', '不礼貌', '冷漠', '不耐烦', '粗暴', '不专业', '不理人', '无礼']
    },
    'Comfort': {
        'positive': ['舒适', '凉爽', '温度适宜', '座位舒服', '安静', '环境好', '干净', '明亮'],
        'negative': ['不舒服', '太冷', '太热', '闷热', '吵闹', '脏', '座位硬', '环境差', '空调太冷', '冷死', '热死']
    },
    'Safety': {
        'positive': ['安全', '放心', '保障', '防护', '稳定', '安心'],
        'negative': ['危险', '不安全', '事故', '摔倒', '受伤', '撞击', '夹到', '滑倒', '碰撞']
    },
    'Service_Facility': {
        'positive': ['设施好', '设备新', '电梯正常', '厕所干净', 'wifi好', '充电方便', '无障碍', '指示清楚'],
        'negative': ['设施差', '设备坏', '电梯故障', '没厕所', '无wifi', '不能充电', '无障碍差', '指示不清']
    }
}

def extract_station_mentions(text):
    """从文本中提取提到的地铁站名"""
    mentioned_stations = []
    for station in STATION_NAMES:
        # 移除可能的特殊字符
        station_clean = station.replace('‎', '').replace('站', '')
        if station_clean in text or (station + '站') in text:
            mentioned_stations.append(station)
    return mentioned_stations

def calculate_sentiment_score(text, keywords):
    """根据关键词计算情感得分"""
    score = 0
    for keyword in keywords:
        # 使用简单的关键词匹配和权重
        count = text.count(keyword)
        if count > 0:
            # 根据关键词的强度给予不同权重
            if len(keyword) >= 4:  # 长关键词权重更高
                score += count * 2
            else:
                score += count
    return score

def analyze_weibo_content(file_path):
    """分析微博内容"""
    # 初始化每个站点和维度的情感得分
    station_scores = defaultdict(lambda: defaultdict(lambda: {'positive': 0, 'negative': 0}))
    
    print("开始分析微博内容...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"已处理 {line_num} 行...")
            
            line = line.strip()
            if not line:
                continue
            
            # 提取提到的地铁站
            mentioned_stations = extract_station_mentions(line)
            
            if mentioned_stations:
                # 对每个服务维度进行分析
                for dimension, keywords in SERVICE_DIMENSIONS.items():
                    pos_score = calculate_sentiment_score(line, keywords['positive'])
                    neg_score = calculate_sentiment_score(line, keywords['negative'])
                    
                    # 为每个提到的站点累加得分
                    for station in mentioned_stations:
                        station_scores[station][dimension]['positive'] += pos_score
                        station_scores[station][dimension]['negative'] += neg_score
    
    print(f"分析完成，共处理 {line_num} 行")
    return station_scores

def generate_csv_files(station_scores):
    """生成12个CSV文件"""
    print("开始生成CSV文件...")
    
    for dimension in SERVICE_DIMENSIONS.keys():
        for sentiment in ['positive', 'negative']:
            filename = f"{dimension.lower()}_{sentiment}_sentiment_scores.csv"
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['name', f'{sentiment}_sentiment_score'])
                
                for station in STATION_NAMES:
                    score = station_scores[station][dimension][sentiment]
                    # 如果得分为0，留空（参考原始CSV格式）
                    score_str = str(score) if score > 0 else ''
                    writer.writerow([station, score_str])
            
            print(f"已生成: {filename}")

def main():
    """主函数"""
    print("=== 深圳地铁服务维度情感分析 ===")
    
    # 分析微博内容
    station_scores = analyze_weibo_content('/Users/leida/TransBert/merged_weibo_content.txt')
    
    # 生成CSV文件
    generate_csv_files(station_scores)
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    total_mentions = sum(len(scores) for scores in station_scores.values())
    print(f"共有 {len(station_scores)} 个站点被提及")
    print(f"总提及次数: {total_mentions}")
    
    # 显示前10个最常被提及的站点
    station_mention_counts = {station: sum(sum(dim.values()) for dim in scores.values()) 
                             for station, scores in station_scores.items()}
    top_stations = sorted(station_mention_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\n前10个最常被提及的站点:")
    for station, count in top_stations:
        print(f"{station}: {count}")
    
    print("\n所有CSV文件已生成完成！")

if __name__ == "__main__":
    main()
