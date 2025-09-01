import osmnx as ox
import os

# Define the place
place = "深圳市, 广东省, 中国"

# Define the tags for subway lines and stations
tags = {"railway": "subway"}

# Extract subway lines
subway_lines = ox.features_from_place(place, tags)

# Define tags for subway stations
station_tags = {"railway": "station", "station": "subway"}

# Extract subway stations
subway_stations = ox.features_from_place(place, station_tags)

# 筛选地理范围内的线路和站点
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Rectangle

# 定义筛选范围（最大纬度22.65，最大经度114.2）
# 这里我们需要定义一个合理的最小范围，假设深圳市的范围
min_lat, max_lat = 22.46, 22.82
min_lon, max_lon = 113.75, 114.43

# 创建边界框
bounds = box(min_lon, min_lat, max_lon, max_lat)

print(f"筛选范围: 纬度 {min_lat}-{max_lat}, 经度 {min_lon}-{max_lon}")
print(f"原始线路数量: {len(subway_lines)}")
print(f"原始站点数量: {len(subway_stations)}")

# 筛选线路和站点
# 确保数据使用正确的坐标系
if subway_lines.crs != 'EPSG:4326':
    subway_lines = subway_lines.to_crs('EPSG:4326')
if subway_stations.crs != 'EPSG:4326':
    subway_stations = subway_stations.to_crs('EPSG:4326')

# 筛选在指定范围内的线路
filtered_lines = subway_lines.cx[min_lon:max_lon, min_lat:max_lat]

# 筛选在指定范围内的站点
filtered_stations = subway_stations.cx[min_lon:max_lon, min_lat:max_lat]
# 去除filtered_stations中name重复的
filtered_stations = filtered_stations.drop_duplicates(subset=['name'])

print(f"筛选后线路数量: {len(filtered_lines)}")
print(f"筛选后站点数量: {len(filtered_stations)}")

# 创建地铁站情感热力图
import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np

# 将站点数据转换为DataFrame并提取坐标
stations_df = filtered_stations.copy()

# 提取经纬度坐标
def extract_coordinates(geometry):
    """从geometry对象中提取经纬度"""
    if geometry.geom_type == 'Point':
        return geometry.y, geometry.x  # 返回 (纬度, 经度)
    else:
        # 如果是其他几何类型，取中心点
        centroid = geometry.centroid
        return centroid.y, centroid.x

# 为每个站点提取坐标
stations_df['lat'] = stations_df['geometry'].apply(lambda x: extract_coordinates(x)[0])
stations_df['lon'] = stations_df['geometry'].apply(lambda x: extract_coordinates(x)[1])

print("站点坐标提取完成:")
print(f"站点数量: {len(stations_df)}")
print(f"坐标范围 - 纬度: {stations_df['lat'].min():.4f} ~ {stations_df['lat'].max():.4f}")
print(f"坐标范围 - 经度: {stations_df['lon'].min():.4f} ~ {stations_df['lon'].max():.4f}")

# 定义service dimensions
service_dimensions = ['safety', 'service_facility', 'comfort', 'staff', 'crowdedness', 'reliability']
sentiment_types = ['negative', 'positive']

# 定义不同情感类型的渐变色
negative_gradient = {
    'gradient': 'rgba(0,0,255,0.3) 0%, rgba(0,255,255,0.5) 25%, rgba(0,255,0,0.7) 50%, rgba(255,255,0,0.8) 75%, rgba(255,0,0,1) 100%',
    'description': 'Negative Sentiment Score'
}

positive_gradient = {
    'gradient': 'rgba(255,0,255,0.3) 0%, rgba(128,0,255,0.5) 25%, rgba(0,0,255,0.7) 50%, rgba(0,255,0,0.8) 75%, rgba(255,255,0,1) 100%',
    'description': 'Positive Sentiment Score'
}

def collect_all_sentiment_scores():
    """收集所有sentiment scores用于统一归一化"""
    all_negative_scores = []
    all_positive_scores = []
    
    print("\n收集所有sentiment scores进行统一归一化...")
    
    for dimension in service_dimensions:
        for sentiment_type in sentiment_types:
            csv_file = f'spatio_distribution/{dimension}_{sentiment_type}_sentiment_scores.csv'
            
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    score_column = f'{sentiment_type}_sentiment_score'
                    
                    if score_column in df.columns:
                        # 替换空值为0，然后筛选有效的scores (> 0)
                        df[score_column] = pd.to_numeric(df[score_column], errors='coerce').fillna(0)
                        valid_scores = df[df[score_column] > 0][score_column]
                        
                        if len(valid_scores) > 0:
                            if sentiment_type == 'negative':
                                all_negative_scores.extend(valid_scores.tolist())
                            else:
                                all_positive_scores.extend(valid_scores.tolist())
                            
                            print(f"  {dimension}_{sentiment_type}: {len(valid_scores)} 个有效数据点, 范围: {valid_scores.min():.1f} - {valid_scores.max():.1f}")
                        else:
                            print(f"  {dimension}_{sentiment_type}: 无有效数据")
                    else:
                        print(f"  {dimension}_{sentiment_type}: 缺少score列")
                        
                except Exception as e:
                    print(f"  {dimension}_{sentiment_type}: 读取错误 - {str(e)}")
            else:
                print(f"  {dimension}_{sentiment_type}: 文件不存在")
    
    print(f"\n=== 统一归一化范围 ===")
    
    negative_min, negative_max = None, None
    if all_negative_scores:
        negative_min = min(all_negative_scores)
        negative_max = max(all_negative_scores)
        print(f"所有negative sentiment: {len(all_negative_scores)} 个数据点")
        print(f"Negative sentiment 全局范围: {negative_min:.1f} - {negative_max:.1f}")
    else:
        print("没有找到negative sentiment数据")
    
    positive_min, positive_max = None, None
    if all_positive_scores:
        positive_min = min(all_positive_scores)
        positive_max = max(all_positive_scores)
        print(f"所有positive sentiment: {len(all_positive_scores)} 个数据点")
        print(f"Positive sentiment 全局范围: {positive_min:.1f} - {positive_max:.1f}")
    else:
        print("没有找到positive sentiment数据")
    
    return {
        'negative': {'min': negative_min, 'max': negative_max},
        'positive': {'min': positive_min, 'max': positive_max}
    }

def create_heatmap_with_unified_normalization(dimension, sentiment_type, data_df, stations_df, filtered_lines, global_ranges):
    """为指定的service dimension和sentiment type创建热力图，使用统一归一化"""

    # 确定使用的渐变色
    if sentiment_type == 'negative':
        gradient_info = negative_gradient
    else:
        gradient_info = positive_gradient
    
    # 合并站点坐标和情感数据
    score_column = f'{sentiment_type}_sentiment_score'
    merged_data = stations_df.merge(data_df, left_on='name', right_on='name', how='left')
    
    # 填充没有情感数据的站点为0
    merged_data[score_column] = merged_data[score_column].fillna(0)
    
    print(f"\n处理 {dimension} - {sentiment_type}:")
    print(f"合并后数据行数: {len(merged_data)}")
    print(f"有情感数据的站点: {len(merged_data[merged_data[score_column] > 0])}")
    
    # 获取该sentiment type的全局范围
    global_min = global_ranges[sentiment_type]['min']
    global_max = global_ranges[sentiment_type]['max']
    
    if global_min is not None and global_max is not None:
        print(f"使用全局归一化范围: {global_min:.1f} - {global_max:.1f}")
        
        # 使用全局范围进行Min-Max归一化
        merged_data['normalized_score'] = 0.0  # 初始化为0
        
        # 只对有效的scores进行归一化
        mask = merged_data[score_column] > 0
        if global_max > global_min:
            merged_data.loc[mask, 'normalized_score'] = (
                merged_data.loc[mask, score_column] - global_min
            ) / (global_max - global_min)
        else:
            # 如果所有scores都相同，设为0.5
            merged_data.loc[mask, 'normalized_score'] = 0.5
        
        # 确保归一化值在[0,1]范围内
        merged_data['normalized_score'] = merged_data['normalized_score'].clip(0, 1)
        
        # 显示当前dimension的scores信息
        valid_scores = merged_data[merged_data[score_column] > 0][score_column]
        if len(valid_scores) > 0:
            print(f"当前维度原始scores范围: {valid_scores.min():.1f} - {valid_scores.max():.1f}")
            valid_normalized = merged_data[merged_data['normalized_score'] > 0]['normalized_score']
            print(f"当前维度归一化后范围: {valid_normalized.min():.3f} - {valid_normalized.max():.3f}")
        else:
            print("当前维度没有有效scores")
        
    else:
        merged_data['normalized_score'] = 0.0
        print(f"无法进行归一化：该sentiment type没有全局数据")
    
    # 检查数据质量
    print(f"最大归一化分数: {merged_data['normalized_score'].max():.3f}")
    print(f"最小归一化分数: {merged_data['normalized_score'].min():.3f}")
    print(f"平均归一化分数: {merged_data['normalized_score'].mean():.3f}")
    
    # 显示分数最高的站点（使用归一化后的值）
    top_stations = merged_data.nlargest(10, 'normalized_score')[['name', 'lat', 'lon', score_column, 'normalized_score']]
    print(f"\n分数最高的10个站点:")
    for idx, row in top_stations.iterrows():
        print(f"{row['name']}: 原始 {row[score_column]:.1f} -> 归一化 {row['normalized_score']:.3f} ({row['lat']:.6f}, {row['lon']:.6f})")
    
    # 创建热力图数据
    # 只包含有情感数据的站点，使用归一化后的scores
    heatmap_data = merged_data[merged_data['normalized_score'] > 0].copy()
    
    # 准备热力图数据 [纬度, 经度, 归一化权重]
    heat_data = []
    for idx, row in heatmap_data.iterrows():
        heat_data.append([row['lat'], row['lon'], row['normalized_score']])
    
    print(f"热力图数据点数量: {len(heat_data)}")
    
    if len(heat_data) > 0:
        print(f"归一化分数范围: {min(point[2] for point in heat_data):.3f} ~ {max(point[2] for point in heat_data):.3f}")
        
        # 创建带站点标记的增强版热力图
        # should use both station_df and filtered_lines to determine the center of the map
        center_lat = stations_df['lat'].mean() + 0.025
        center_lon = stations_df['lon'].mean() + 0.035
        
        # 创建基础地图
        m_enhanced = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='CartoDB Positron',
            control_scale=True,
            zoom_control=False
        )
        # 绘制地铁线路
        for idx, row in filtered_lines.iterrows():
            locations = [[point[1], point[0]] for point in row['geometry'].coords]
            folium.PolyLine(
                locations=locations,
                color='grey',
                weight=1,
                opacity=0.5
            ).add_to(m_enhanced)
        
        # 添加热力图层（使用归一化后的数据）
        gradient_1 = {.4: "blue", .6: "cyan", .7: "lime", .8: "yellow", 1: "red"}
        HeatMap(
            heat_data,
            min_opacity=0.3,
            radius=25,
            blur=15,
            max_zoom=18,
            gradient=gradient_1
        ).add_to(m_enhanced)

        compass_html = """
        <div style="position: fixed; 
                    top: 5px; right: 5px; width: 50px; height: 70px; 
                    background-color: transparent; z-index:9999; 
                    display: flex; flex-direction: column; align-items: center; justify-content: center;
                    padding: 5px;
                    ">
        <div style="font-size: 20px; font-weight: bold; color: #333; margin-bottom: 1px;">N</div>
        <svg width="36" height="36" viewBox="0 0 24 24" style="margin-top: 1px;">
            <path d="M12 2 L20 20 L12 16 L4 20 Z" 
                  fill="none" 
                  stroke="#333" 
                  stroke-width="2" 
                  stroke-linejoin="round"/>
        </svg>
        </div>
        """
        m_enhanced.get_root().html.add_child(folium.Element(compass_html))

        # 添加综合图例（显示统一归一化信息）
        # to two columns, one for Sentiment Score Gradient, and one for Metro Infrastructure Legend
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 1px; right: 1px; width: 300px; height: 134px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:13px; padding: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.3);
                    border-radius: 5px;
                    ">
        
        <div style="display: flex; gap: 3px;">
            <!-- Left Column: Sentiment Score Gradient -->
            <div style="flex: 1;">
                <h5 style="margin:0 0 8px 0; font-size:13px;">{"Negative" if sentiment_type == "negative" else "Positive"} Sentiment Score</h5>
                <div style="display: flex; align-items: center;">
                    <div style="background: linear-gradient(to top, 
                                rgba(0,0,255,0.3) 0%, 
                                rgba(0,255,255,0.5) 25%, 
                                rgba(0,255,0,0.7) 50%, 
                                rgba(255,255,0,0.8) 75%, 
                                rgba(255,0,0,1) 100%); 
                                width: 20px; height: 80px; margin-right: 8px; border: 1px solid #ccc;">
                    </div>
                    <div style="height: 80px; display: flex; flex-direction: column; justify-content: space-between; font-size:10px;">
                        <span>1.000 (High)</span>
                        <span>0.750</span>
                        <span>0.500</span>
                        <span>0.250</span>
                        <span>0.000 (Low)</span>
                    </div>
                </div>
            </div>
            
            <!-- Right Column: Metro Infrastructure Legend -->
            <div style="flex: 1;">
                <h5 style="margin:0 0 8px 0; font-size:13px;">Metro Infrastructure</h5>
                
                <!-- Subway Lines -->
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 20px; height: 2px; background-color: grey; margin-right: 8px; opacity: 0.5;"></div>
                    <span style="font-size:10px;">Metro Lines</span>
                </div>
                
                <!-- Subway Stations with Data -->
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 8px; height: 8px; border: 2px solid black; border-radius: 50%; 
                                background-color: transparent; margin-right: 10px;"></div>
                    <span style="font-size:10px;">Stations (with sentiment data)</span>
                </div>
                
                <!-- Subway Stations without Data -->
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="width: 8px; height: 8px; border: 2px solid gray; border-radius: 50%; 
                                background-color: transparent; margin-right: 10px;"></div>
                    <span style="font-size:10px;">Stations (no sentiment data)</span>
                </div>
            </div>
        </div>
        
        </div>
        """
        m_enhanced.get_root().html.add_child(folium.Element(legend_html))
        
        # 添加所有地铁站点标记
        for idx, row in merged_data.iterrows():
            # 根据归一化后的分数选择颜色和大小
            if row['normalized_score'] > 0:
                color = 'black'
                radius = 2
                station_type = "with sentiment data"
            else:
                # 没有情感数据的站点用灰色
                color = 'gray'
                radius = 2
                station_type = "no sentiment data"
            
            # 创建弹窗文本（显示原始值、归一化值和全局范围信息）
            popup_text = f"""
            <div style="min-width: 220px;">
                <h4 style="margin-top:0; margin-bottom:8px; color:#333;">{row['name']}</h4>
                <table style="font-size:12px; width:100%;">
                    <tr><td><strong>Type:</strong></td><td>Subway Station ({station_type})</td></tr>
                    <tr><td><strong>Original Score:</strong></td><td>{row[score_column]:.1f}</td></tr>
                    <tr><td><strong>Normalized Score:</strong></td><td>{row['normalized_score']:.3f}</td></tr>
                    <tr><td><strong>Global Range:</strong></td><td>{global_min:.1f} - {global_max:.1f}</td></tr>
                    <tr><td><strong>Latitude:</strong></td><td>{row['lat']:.6f}</td></tr>
                    <tr><td><strong>Longitude:</strong></td><td>{row['lon']:.6f}</td></tr>
                </table>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.,
                weight=2
            ).add_to(m_enhanced)
        
        # 保存增强版地图（添加统一归一化标识）
        output_file = f'shenzhen_metro_{dimension}_{sentiment_type}_sentiment_unified_normalized_heatmap.html'
        m_enhanced.save(output_file)
        print(f"统一归一化热力图已保存为: {output_file}")
        
        return m_enhanced
    else:
        print(f"无法创建 {dimension} - {sentiment_type} 地图：没有有效的数据点")
        return None

# 主处理流程
print("\n开始处理所有service dimensions和sentiment types (统一归一化)...")

# 第一步：收集所有sentiment scores计算全局范围
global_ranges = collect_all_sentiment_scores()

# 第二步：使用全局范围创建统一归一化的热力图
for dimension in service_dimensions:
    print(f"\n{'='*50}")
    print(f"处理Service Dimension: {dimension}")
    print(f"{'='*50}")
    
    for sentiment_type in sentiment_types:
        print(f"\n{'-'*30}")
        print(f"处理Sentiment Type: {sentiment_type}")
        print(f"{'-'*30}")
        
        # 读取对应的CSV文件
        csv_file = f'spatio_distribution/{dimension}_{sentiment_type}_sentiment_scores.csv'
        
        if os.path.exists(csv_file):
            try:
                df_sentiment = pd.read_csv(csv_file)
                # 处理空值和数据类型
                score_column = f'{sentiment_type}_sentiment_score'
                if score_column in df_sentiment.columns:
                    df_sentiment[score_column] = pd.to_numeric(df_sentiment[score_column], errors='coerce').fillna(0)
                
                print(f"成功读取文件: {csv_file}")
                print(f"数据行数: {len(df_sentiment)}")
                
                # 创建使用统一归一化的热力图
                heatmap = create_heatmap_with_unified_normalization(
                    dimension, sentiment_type, df_sentiment, stations_df, filtered_lines, global_ranges
                )
                
                if heatmap is not None:
                    print(f"✓ 成功创建 {dimension} - {sentiment_type} 统一归一化热力图")
                else:
                    print(f"✗ 创建 {dimension} - {sentiment_type} 统一归一化热力图失败")
                
            except Exception as e:
                print(f"✗ 处理 {csv_file} 时出错: {str(e)}")
        else:
            print(f"✗ 文件不存在: {csv_file}")

print("\n" + "="*60)
print("所有统一归一化热力图处理完成！")
print("="*60)
print("\n重要说明:")
print("- 所有negative sentiment的CSV文件使用统一的归一化范围")
print("- 所有positive sentiment的CSV文件使用统一的归一化范围")
print("- 这样可以保持不同CSV文件之间数据差异的可比较性")
print("- 热力图文件名包含'unified_normalized'标识")
