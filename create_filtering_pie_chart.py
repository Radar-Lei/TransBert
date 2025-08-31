import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体支持
plt.rcParams["svg.fonttype"] = "none"

# 数据
total_samples = 512738
transit_related = 116506
non_transit = total_samples - transit_related

# 饼图数据
labels = ['Transit-related posts', 'Non-transit posts']
sizes = [transit_related, non_transit]
colors = ['#66b3ff', '#ff9999']
explode = (0.1, 0)  # 突出显示交通相关帖子

# 创建图表
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                  autopct='%1.1f%%', shadow=True, startangle=90,
                                  textprops={'fontsize': 12})

# 设置百分比文本样式
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

# 确保饼图是圆形
ax.axis('equal')

# 添加图例
ax.legend(wedges, [f'{label}: {size:,}' for label, size in zip(labels, sizes)],
          title="Post Categories",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout()

# 保存为SVG格式
plt.savefig('filtering_results_pie_chart.svg', format='svg', bbox_inches='tight', dpi=300)

print("饼图已生成并保存为 'filtering_results_pie_chart.svg' 和 'filtering_results_pie_chart.png'")
print(f"总样本数: {total_samples:,}")
print(f"交通相关帖子: {transit_related:,} ({transit_related/total_samples*100:.1f}%)")
print(f"非交通相关帖子: {non_transit:,} ({non_transit/total_samples*100:.1f}%)")

plt.show()