
import osmnx as ox
import os
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Define the place
place = "Shenzhen, Guangdong, China"

# Define the tags for subway lines and stations
tags = {"railway": "subway"}

# Extract subway lines
subway_lines = ox.features_from_place(place, tags)

# Define tags for subway stations
station_tags = {"railway": "station", "station": "subway"}

# Extract subway stations
subway_stations = ox.features_from_place(place, station_tags)

# Filter geographic range
min_lat, max_lat = 22.46, 22.82
min_lon, max_lon = 113.75, 114.43

# Create bounding box
bounds = box(min_lon, min_lat, max_lon, max_lat)

print(f"Filter range: Latitude {min_lat}-{max_lat}, Longitude {min_lon}-{max_lon}")
print(f"Original lines count: {len(subway_lines)}")
print(f"Original stations count: {len(subway_stations)}")

# Filter lines and stations
if subway_lines.crs != 'EPSG:4326':
    subway_lines = subway_lines.to_crs('EPSG:4326')
if subway_stations.crs != 'EPSG:4326':
    subway_stations = subway_stations.to_crs('EPSG:4326')

# Filter within specified range
filtered_lines = subway_lines.cx[min_lon:max_lon, min_lat:max_lat]
filtered_stations = subway_stations.cx[min_lon:max_lon, min_lat:max_lat]
filtered_stations = filtered_stations.drop_duplicates(subset=['name'])

print(f"Filtered lines count: {len(filtered_lines)}")
print(f"Filtered stations count: {len(filtered_stations)}")

# Convert station data to DataFrame and extract coordinates
stations_df = filtered_stations.copy()

def extract_coordinates(geometry):
    """Extract coordinates from geometry object"""
    if geometry.geom_type == 'Point':
        return geometry.y, geometry.x  # Return (latitude, longitude)
    else:
        centroid = geometry.centroid
        return centroid.y, centroid.x

# Extract coordinates for each station
stations_df['lat'] = stations_df['geometry'].apply(lambda x: extract_coordinates(x)[0])
stations_df['lon'] = stations_df['geometry'].apply(lambda x: extract_coordinates(x)[1])

print("Station coordinates extraction completed:")
print(f"Station count: {len(stations_df)}")
print(f"Coordinate range - Latitude: {stations_df['lat'].min():.4f} ~ {stations_df['lat'].max():.4f}")
print(f"Coordinate range - Longitude: {stations_df['lon'].min():.4f} ~ {stations_df['lon'].max():.4f}")

# Define service dimensions
service_dimensions = ['safety', 'service_facility', 'comfort', 'staff', 'crowdedness', 'reliability']
sentiment_types = ['negative', 'positive']

# Get RdYlGn colormap
rdylgn = plt.cm.RdYlGn

# Define positions
positions = [0.4, 0.6, 0.7, 0.8, 1.0]

# Create two gradients
gradient1 = {}
gradient2 = {}

for pos in positions:
    # First gradient: from left side of RdYlGn (negative) - reversed order
    # Use positions in reverse order to reverse the color gradient
    reversed_pos = 1.0 - pos
    color1 = rdylgn(reversed_pos * 0.5)  # Use first half of colormap in reverse
    gradient1[pos] = f"rgb({int(color1[0]*255)}, {int(color1[1]*255)}, {int(color1[2]*255)})"
    
    # Second gradient: from right side of RdYlGn (positive)
    color2 = rdylgn(0.5 + pos * 0.5)  # Use second half of colormap
    gradient2[pos] = f"rgb({int(color2[0]*255)}, {int(color2[1]*255)}, {int(color2[2]*255)})"

print("Gradient 1 (Negative):", gradient1)
print("Gradient 2 (Positive):", gradient2)

# Extract RGB values for legend
gradient1_colors = list(gradient1.values())
gradient2_colors = list(gradient2.values())

print("Gradient 1 Colors:", gradient1_colors)
print("Gradient 2 Colors:", gradient2_colors)

def collect_all_sentiment_scores():
    """Collect all sentiment scores for unified normalization"""
    all_negative_scores = []
    all_positive_scores = []
    
    print("\nCollecting all sentiment scores for unified normalization...")
    
    for dimension in service_dimensions:
        for sentiment_type in sentiment_types:
            csv_file = f'spatio_distribution/{dimension}_{sentiment_type}_sentiment_scores.csv'
            
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    score_column = f'{sentiment_type}_sentiment_score'
                    
                    if score_column in df.columns:
                        df[score_column] = pd.to_numeric(df[score_column], errors='coerce').fillna(0)
                        valid_scores = df[df[score_column] > 0][score_column]
                        
                        if len(valid_scores) > 0:
                            if sentiment_type == 'negative':
                                all_negative_scores.extend(valid_scores.tolist())
                            else:
                                all_positive_scores.extend(valid_scores.tolist())
                            
                            print(f"  {dimension}_{sentiment_type}: {len(valid_scores)} valid data points, range: {valid_scores.min():.1f} - {valid_scores.max():.1f}")
                        else:
                            print(f"  {dimension}_{sentiment_type}: No valid data")
                    else:
                        print(f"  {dimension}_{sentiment_type}: Missing score column")
                        
                except Exception as e:
                    print(f"  {dimension}_{sentiment_type}: Read error - {str(e)}")
            else:
                print(f"  {dimension}_{sentiment_type}: File does not exist")
    
    print(f"\n=== Unified Normalization Range ===")
    
    negative_min, negative_max = None, None
    if all_negative_scores:
        negative_min = min(all_negative_scores)
        negative_max = max(all_negative_scores)
        print(f"All negative sentiment: {len(all_negative_scores)} data points")
        print(f"Negative sentiment global range: {negative_min:.1f} - {negative_max:.1f}")
    else:
        print("No negative sentiment data found")
    
    positive_min, positive_max = None, None
    if all_positive_scores:
        positive_min = min(all_positive_scores)
        positive_max = max(all_positive_scores)
        print(f"All positive sentiment: {len(all_positive_scores)} data points")
        print(f"Positive sentiment global range: {positive_min:.1f} - {positive_max:.1f}")
    else:
        print("No positive sentiment data found")
    
    return {
        'negative': {'min': negative_min, 'max': negative_max},
        'positive': {'min': positive_min, 'max': positive_max}
    }

def create_combined_heatmap(sentiment_type, all_data_df, stations_df, filtered_lines, global_ranges):
    """Create heatmap combining all service dimensions for a specific sentiment type"""
    
    print(f"\nProcessing {sentiment_type.capitalize()} Sentiment - Combined Heatmap:")
    
    # Merge station coordinates with sentiment data
    merged_data = stations_df.merge(all_data_df, left_on='name', right_on='name', how='left')
    
    # Fill missing sentiment data with 0
    score_column = f'{sentiment_type}_sentiment_score'
    merged_data[score_column] = merged_data[score_column].fillna(0)
    
    print(f"Merged data rows: {len(merged_data)}")
    print(f"Stations with sentiment data: {len(merged_data[merged_data[score_column] > 0])}")
    
    # Get global range
    global_min = global_ranges[sentiment_type]['min']
    global_max = global_ranges[sentiment_type]['max']
    
    # Normalization processing
    if global_min is not None and global_max is not None:
        print(f"Using global normalization range: {global_min:.1f} - {global_max:.1f}")
        
        merged_data['normalized_score'] = 0.0
        mask = merged_data[score_column] > 0
        
        if global_max > global_min:
            merged_data.loc[mask, 'normalized_score'] = (
                merged_data.loc[mask, score_column] - global_min
            ) / (global_max - global_min)
        else:
            merged_data.loc[mask, 'normalized_score'] = 0.5
        
        merged_data['normalized_score'] = merged_data['normalized_score'].clip(0, 1)
        
        # Display current scores information
        valid_scores = merged_data[merged_data[score_column] > 0][score_column]
        if len(valid_scores) > 0:
            print(f"Original scores range: {valid_scores.min():.1f} - {valid_scores.max():.1f}")
            valid_normalized = merged_data[merged_data['normalized_score'] > 0]['normalized_score']
            print(f"Normalized range: {valid_normalized.min():.3f} - {valid_normalized.max():.3f}")
        else:
            print("No valid scores found")
    else:
        merged_data['normalized_score'] = 0.0
        print(f"Cannot normalize: No global data for {sentiment_type} sentiment")
    
    # Check data quality
    print(f"Max normalized score: {merged_data['normalized_score'].max():.3f}")
    print(f"Min normalized score: {merged_data['normalized_score'].min():.3f}")
    print(f"Avg normalized score: {merged_data['normalized_score'].mean():.3f}")
    
    # Prepare heatmap data
    heatmap_data = merged_data[merged_data['normalized_score'] > 0].copy()
    heat_data = []
    
    for idx, row in heatmap_data.iterrows():
        heat_data.append([row['lat'], row['lon'], row['normalized_score']])
    
    print(f"Heatmap data points: {len(heat_data)}")
    
    if len(heat_data) > 0:
        print(f"Normalized score range: {min(point[2] for point in heat_data):.3f} ~ {max(point[2] for point in heat_data):.3f}")
        
        # Create base map
        center_lat = stations_df['lat'].mean() + 0.025
        center_lon = stations_df['lon'].mean() + 0.035
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='CartoDB Positron',
            control_scale=True,
            zoom_control=False
        )
        
        # Draw subway lines
        for idx, row in filtered_lines.iterrows():
            locations = [[point[1], point[0]] for point in row['geometry'].coords]
            folium.PolyLine(
                locations=locations,
                color='grey',
                weight=1,
                opacity=0.5
            ).add_to(m)
        
        # Add heatmap layer with correct gradient
        if sentiment_type == 'negative':
            # Use gradient1 for negative sentiment
            HeatMap(
                heat_data,
                min_opacity=0.3,
                radius=25,
                blur=15,
                max_zoom=18,
                gradient=gradient1
            ).add_to(m)
        else:
            # Use gradient2 for positive sentiment
            HeatMap(
                heat_data,
                min_opacity=0.3,
                radius=25,
                blur=15,
                max_zoom=18,
                gradient=gradient2
            ).add_to(m)
        
        # Add compass
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
        m.get_root().html.add_child(folium.Element(compass_html))
        
        # Add comprehensive legend with correct colorbar for each sentiment type
        sentiment_title = "Negative" if sentiment_type == "negative" else "Positive"
        
        # Define color gradients for legend using exact gradient values
        if sentiment_type == "negative":
            # Use gradient1 colors for negative sentiment
            color_gradient = f"""
            background: linear-gradient(to top,
                        {gradient1_colors[0].replace('rgb', 'rgba').replace(')', ',0.3)')} 0%,
                        {gradient1_colors[1].replace('rgb', 'rgba').replace(')', ',0.5)')} 25%,
                        {gradient1_colors[2].replace('rgb', 'rgba').replace(')', ',0.7)')} 50%,
                        {gradient1_colors[3].replace('rgb', 'rgba').replace(')', ',0.8)')} 75%,
                        {gradient1_colors[4].replace('rgb', 'rgba').replace(')', ',1)')} 100%);
            """
        else:
            # Use gradient2 colors for positive sentiment
            color_gradient = f"""
            background: linear-gradient(to top,
                        {gradient2_colors[0].replace('rgb', 'rgba').replace(')', ',0.3)')} 0%,
                        {gradient2_colors[1].replace('rgb', 'rgba').replace(')', ',0.5)')} 25%,
                        {gradient2_colors[2].replace('rgb', 'rgba').replace(')', ',0.7)')} 50%,
                        {gradient2_colors[3].replace('rgb', 'rgba').replace(')', ',0.8)')} 75%,
                        {gradient2_colors[4].replace('rgb', 'rgba').replace(')', ',1)')} 100%);
            """
        
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
                <h5 style="margin:0 0 8px 0; font-size:13px;">{sentiment_title} Sentiment Score</h5>
                <div style="display: flex; align-items: center;">
                    <div style="{color_gradient}
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
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add all subway station markers
        for idx, row in merged_data.iterrows():
            if row['normalized_score'] > 0:
                color = 'black'
                radius = 2
                station_type = "with sentiment data"
            else:
                color = 'gray'
                radius = 2
                station_type = "no sentiment data"
            
            # Create popup text
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
                fillOpacity=0.7,
                weight=1
            ).add_to(m)
        
        # Save enhanced map
        output_file = f'shenzhen_metro_combined_{sentiment_type}_sentiment_unified_normalized_heatmap.html'
        m.save(output_file)
        print(f"Combined heatmap saved as: {output_file}")
        
        return m
    else:
        print(f"Cannot create {sentiment_type} combined heatmap: No valid data points")
        return None

# Main processing flow
print("\nStarting processing of combined sentiment heatmaps (unified normalization)...")

# Step 1: Collect all sentiment scores to calculate global ranges
global_ranges = collect_all_sentiment_scores()

# Step 2: Combine data from all service dimensions for each sentiment type
for sentiment_type in sentiment_types:
    print(f"\n{'='*50}")
    print(f"Processing {sentiment_type.capitalize()} Sentiment - Combined Data")
    print(f"{'='*50}")
    
    combined_data = pd.DataFrame(columns=['name', f'{sentiment_type}_sentiment_score'])
    
    for dimension in service_dimensions:
        csv_file = f'spatio_distribution/{dimension}_{sentiment_type}_sentiment_scores.csv'
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                score_column = f'{sentiment_type}_sentiment_score'
                
                if score_column in df.columns:
                    df[score_column] = pd.to_numeric(df[score_column], errors='coerce').fillna(0)
                    
                    # Combine data by taking maximum score for each station
                    temp_combined = combined_data.merge(df[['name', score_column]], on='name', how='outer')
                    
                    # For stations with multiple scores, take the maximum
                    if f'{sentiment_type}_sentiment_score_x' in temp_combined.columns:
                        temp_combined[score_column] = temp_combined[[f'{sentiment_type}_sentiment_score_x', f'{sentiment_type}_sentiment_score_y']].max(axis=1)
                        temp_combined = temp_combined.drop([f'{sentiment_type}_sentiment_score_x', f'{sentiment_type}_sentiment_score_y'], axis=1)
                    
                    combined_data = temp_combined
                    print(f"  Added data from {dimension}: {len(df[df[score_column] > 0])} valid data points")
                    
                else:
                    print(f"  {dimension}: Missing score column")
                    
            except Exception as e:
                print(f"  {dimension}: Read error - {str(e)}")
        else:
            print(f"  {dimension}: File does not exist")
    
    print(f"Combined {sentiment_type} data: {len(combined_data)} stations")
    print(f"Stations with valid {sentiment_type} scores: {len(combined_data[combined_data[f'{sentiment_type}_sentiment_score'] > 0])}")
    
    # Create combined heatmap
    heatmap = create_combined_heatmap(sentiment_type, combined_data, stations_df, filtered_lines, global_ranges)
    
    if heatmap is not None:
        print(f"✓ Successfully created combined {sentiment_type} heatmap")
    else:
        print(f"✗ Failed to create combined {sentiment_type} heatmap")

print("\n" + "="*60)
print("All combined heatmap processing completed!")
print("="*60)
print("\nImportant Notes:")
print("- Two HTML files created: one for negative sentiment, one for positive sentiment")
print("- Data combined from all service dimensions (safety, service_facility, comfort, staff, crowdedness, reliability)")
print("- Negative sentiment uses RdYlGn colormap first half gradient")
print("- Positive sentiment uses RdYlGn colormap second half gradient")
print("- All data uses unified normalization range for comparability")
print("- All text in English language")