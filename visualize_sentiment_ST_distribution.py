import osmnx as ox

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

# 创建地铁站繁忙度热力图
import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np

# 读取站点繁忙度数据
df_station_rank = pd.read_csv('station_sentiment_rank.csv')
df_station_rank = df_station_rank.fillna(0)


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

# 显示前几个站点的信息
print("\n前5个站点的坐标:")
for idx, row in stations_df.head(5).iterrows():
    print(f"{row['name']}: ({row['lat']:.6f}, {row['lon']:.6f})")

# 合并站点坐标和繁忙度数据
merged_data = stations_df.merge(df_station_rank, left_on='name', right_on='name', how='left')

# 填充没有繁忙度数据的站点为0
merged_data['negative_sentiment_score'] = merged_data['negative_sentiment_score'].fillna(0)

print("数据合并完成:")
print(f"合并后数据行数: {len(merged_data)}")
print(f"有繁忙度数据的站点: {len(merged_data[merged_data['negative_sentiment_score'] > 0])}")

# 对scores进行归一化到[0,1]区间
# 只对有数据的站点进行归一化（score > 0）
valid_scores = merged_data[merged_data['negative_sentiment_score'] > 0]['negative_sentiment_score']

if len(valid_scores) > 0:
    min_score = valid_scores.min()
    max_score = valid_scores.max()
    
    print(f"\n原始scores范围: {min_score:.2f} ~ {max_score:.2f}")
    
    # 使用Min-Max归一化
    merged_data['normalized_score'] = 0.0  # 初始化为0
    
    # 只对有效的scores进行归一化
    mask = merged_data['negative_sentiment_score'] > 0
    if max_score > min_score:
        merged_data.loc[mask, 'normalized_score'] = (
            merged_data.loc[mask, 'negative_sentiment_score'] - min_score
        ) / (max_score - min_score)
    else:
        # 如果所有scores都相同，设为0.5
        merged_data.loc[mask, 'normalized_score'] = 0.5
    
    print(f"归一化后scores范围: {merged_data['normalized_score'].min():.3f} ~ {merged_data['normalized_score'].max():.3f}")
else:
    merged_data['normalized_score'] = 0.0
    print("没有有效的scores进行归一化")

# 检查数据质量
print("\n归一化后的繁忙度分布:")
print(f"最大归一化繁忙度: {merged_data['normalized_score'].max():.3f}")
print(f"最小归一化繁忙度: {merged_data['normalized_score'].min():.3f}")
print(f"平均归一化繁忙度: {merged_data['normalized_score'].mean():.3f}")

# 显示繁忙度最高的站点（使用归一化后的值）
top_busy_stations = merged_data.nlargest(10, 'normalized_score')[['name', 'lat', 'lon', 'negative_sentiment_score', 'normalized_score']]
print("\n繁忙度最高的10个站点:")
for idx, row in top_busy_stations.iterrows():
    print(f"{row['name']}: 原始 {row['negative_sentiment_score']:.1f} -> 归一化 {row['normalized_score']:.3f} ({row['lat']:.6f}, {row['lon']:.6f})")

# 创建热力图数据
# 只包含有繁忙度数据的站点，使用归一化后的scores
heatmap_data = merged_data[merged_data['normalized_score'] > 0].copy()

# 准备热力图数据 [纬度, 经度, 归一化权重]
heat_data = []
for idx, row in heatmap_data.iterrows():
    heat_data.append([row['lat'], row['lon'], row['normalized_score']])

print(f"\n热力图数据点数量: {len(heat_data)}")

if len(heat_data) > 0:
    print(f"归一化繁忙度范围: {min(point[2] for point in heat_data):.3f} ~ {max(point[2] for point in heat_data):.3f}")
    
    # 显示前几个数据点
    print("\n前5个热力图数据点:")
    for i, point in enumerate(heat_data[:5]):
        print(f"  {i+1}. 纬度: {point[0]:.6f}, 经度: {point[1]:.6f}, 归一化繁忙度: {point[2]:.3f}")
else:
    print("没有找到有效的热力图数据点")

# 创建带站点标记的增强版热力图
center_lat = merged_data['lat'].mean()
center_lon = merged_data['lon'].mean()

if len(heat_data) > 0:
    # 创建基础地图
    m_enhanced = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='CartoDB Positron',
        control_scale=True,
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
    HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=25,
        blur=15,
        max_zoom=18
    ).add_to(m_enhanced)
    
    # 添加综合图例（包含热力图、地铁线路和站点）
    legend_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 220px; height: 280px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:13px; padding: 12px; box-shadow: 0 0 15px rgba(0,0,0,0.3);
                border-radius: 5px;
                ">
    <!-- Heat Map Legend -->
    <h4 style="margin-top:0; margin-bottom:12px; font-size:15px; text-align:center;">
        Map Legend
    </h4>
    
    <!-- Sentiment Score Gradient -->
    <div style="margin-bottom: 15px;">
        <h5 style="margin:0 0 8px 0; font-size:13px;">Negative Sentiment Score</h5>
        <div style="display: flex; align-items: center;">
            <div style="background: linear-gradient(to top, 
                        rgba(0,0,255,0.3) 0%, 
                        rgba(0,255,255,0.5) 25%, 
                        rgba(0,255,0,0.7) 50%, 
                        rgba(255,255,0,0.8) 75%, 
                        rgba(255,0,0,1) 100%); 
                        width: 20px; height: 60px; margin-right: 8px; border: 1px solid #ccc;">
            </div>
            <div style="height: 60px; display: flex; flex-direction: column; justify-content: space-between; font-size:11px;">
                <span>1.000 (High)</span>
                <span>0.750</span>
                <span>0.500</span>
                <span>0.250</span>
                <span>0.000 (Low)</span>
            </div>
        </div>
    </div>
    
    <!-- Subway Infrastructure Legend -->
    <div style="margin-bottom: 10px;">
        <h5 style="margin:0 0 8px 0; font-size:13px;">Subway Infrastructure</h5>
        
        <!-- Subway Lines -->
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 20px; height: 2px; background-color: grey; margin-right: 8px; opacity: 0.5;"></div>
            <span style="font-size:11px;">Subway Lines</span>
        </div>
        
        <!-- Subway Stations with Data -->
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 8px; height: 8px; border: 2px solid black; border-radius: 50%; 
                        background-color: transparent; margin-right: 10px;"></div>
            <span style="font-size:11px;">Stations (with sentiment data)</span>
        </div>
        
        <!-- Subway Stations without Data -->
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 8px; height: 8px; border: 2px solid gray; border-radius: 50%; 
                        background-color: transparent; margin-right: 10px;"></div>
            <span style="font-size:11px;">Stations (no sentiment data)</span>
        </div>
    </div>
    
    <!-- Note -->
    <div style="font-size: 10px; color: #666; border-top: 1px solid #eee; padding-top: 8px; margin-top: 8px;">
        <strong>Note:</strong> Heat map shows normalized sentiment scores. 
        Higher values indicate more negative sentiment at that location.
    </div>
    </div>
    """
    m_enhanced.get_root().html.add_child(folium.Element(legend_html))
    
    # 添加所有地铁站点标记
    for idx, row in merged_data.iterrows():
        # 根据归一化后的繁忙度选择颜色和大小
        if row['normalized_score'] > 0:
            color = 'black'
            radius = 2
            station_type = "with sentiment data"
        else:
            # 没有繁忙度数据的站点用灰色
            color = 'gray'
            radius = 2
            station_type = "no sentiment data"
        
        # 创建弹窗文本（显示原始值和归一化值）
        popup_text = f"""
        <div style="min-width: 200px;">
            <h4 style="margin-top:0; margin-bottom:8px; color:#333;">{row['name']}</h4>
            <table style="font-size:12px; width:100%;">
                <tr><td><strong>Type:</strong></td><td>Subway Station ({station_type})</td></tr>
                <tr><td><strong>Original Score:</strong></td><td>{row['negative_sentiment_score']:.1f}</td></tr>
                <tr><td><strong>Normalized Score:</strong></td><td>{row['normalized_score']:.3f}</td></tr>
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
    
    # 保存增强版地图
    enhanced_map_file = 'shenzhen_metro_sentiment_heatmap_with_legend.html'
    m_enhanced.save(enhanced_map_file)
    print(f"\n增强版热力图已保存为: {enhanced_map_file}")
    
    # 显示增强版地图
    m_enhanced
else:
    print("无法创建增强版地图：没有有效的数据点")