import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from matplotlib.lines import Line2D

figure_colors = ['#FF9494', '#FFCFB3', '#FFF5E4', '#B7E0FF', '#DFF2EB', '#4A628A', '#AA96DA', '#FCBAD3']
font_path = '/data/ljc/.local/share/fonts/Times New Roman.ttf'  # 你需要更改为字体的实际路径
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def set_iclr_style(layout=3):

    if layout == 2:
        figsize = (7, 3.5)
        base_font = 10
    elif layout == 3:
        figsize = (7, 5)
        base_font = 12
    elif layout == 4:
        figsize = (7, 7)
        base_font = 14   # 四栏图加大字体
    else:
        raise ValueError("layout 必须是 'double' | 'triple' | 'quad'")

    plt.rcParams.update({
        "figure.figsize": figsize,

        # 字体大小（随布局调整）
        "font.size": base_font,
        "axes.labelsize": base_font + 3,
        "legend.fontsize": base_font + 1,
        "xtick.labelsize": base_font + 2,
        "ytick.labelsize": base_font + 2,

        # 线条
        # "lines.linewidth": 2.75,
        # "lines.markersize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        # 输出质量
        "figure.dpi": 300,

        # 字体族
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
    })
    print(f"[ICLR style applied] layout={layout}, figsize={figsize}, base_font={base_font}")

def prepare_fig_input(data, label_ls):
    data_dic = defaultdict(list)
    for k1, v1 in data.items():
        if len(label_ls) == 2:
            data_dic[label_ls[0]].append(k1)
            data_dic[label_ls[1]].append(v1)
        else:
            for k2, v2 in v1.items():
                data_dic[label_ls[0]].append(k1)
                data_dic[label_ls[1]].append(v2)
                data_dic[label_ls[2]].append(k2)
    figure_input = pd.DataFrame(data_dic)
    return figure_input

def draw_bar(data, path):
   
    names = list(data.columns)
    sns.set_theme(style='whitegrid')
    set_iclr_style(layout=3)
    plt.figure()
    colors = sns.color_palette(figure_colors)
    
    ax = sns.barplot(x=names[0], 
                    y=names[1], 
                    hue=names[2] if len(names) > 2 else None, 
                    data=data,  
                    edgecolor='black',
                    palette=colors,
                    width=0.8)
    
    type_num = len(data[names[0]].unique())
    hue_num = len(data[names[2]].unique()) if len(names)> 2 else None 
    # text_size = 20 if hue_num == 2 else 16
    for i, bar in enumerate(ax.patches):
        if hue_num and i >= len(ax.patches) - hue_num:
            break
        bar_height = bar.get_height()
        category = data[names[0]].unique()[i % type_num]  # 每个类别有3个条形图
        hue_value = data[names[2]].unique()[i // type_num]  # 获取当前条形的hue值（low, mid, high）

        # 获取实际的数值，这样我们可以为每个条形图添加不同的文本
        value = data[(data[names[0]] == category) & (data[names[2]] == hue_value)][names[1]].values[0]
        # 在条形顶部稍微向上偏移一点
        # i//3 确保每组type的条形图正确处理
        ax.text(bar.get_x() + bar.get_width() / 2, bar_height * 1.01,  # X轴位置调整到每个条形的中心
                f'{value:.2f}', ha='center', va='bottom')
    ax.legend(loc='best', fancybox=True, framealpha=0.5)  # 图例
    ax.set_xlabel(ax.get_xlabel())  # X轴标签
    ax.set_ylabel(ax.get_ylabel())  # Y轴标签

  
    # ax.tick_params(axis='both', which='major', labelsize=24)
    # plt.ylim(0.2, 0.4)
    
    plt.subplots_adjust(left=0.10, right=0.97, top=0.99, bottom=0.11)
    plt.show()
    plt.savefig(path)
    plt.close()
    
    
def draw_scatter(data, path):
    names = list(data.columns)
    sns.set_theme(style='whitegrid')
    set_iclr_style(layout=3)
    plt.figure()
    colors = sns.color_palette(figure_colors)
    markers = ['o', 's', '^', 'v', 'D', 'P', '*']
    for i in range(len(data)):
        ax = sns.scatterplot(x=[data[names[0]][i]], y=[data[names[1]][i]], 
                        marker=markers[i % len(markers)], color=colors[i % len(colors)], edgecolor='black')
        plt.text(data[names[0]][i]*1.1, data[names[1]][i], data[names[2]][i], ha='left', va='center')

    # # 设置标题和标签
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    # ax.tick_params(axis='both', which='major', labelsize=20)
    plt.xscale('log')
    plt.xticks([10**2, 250, 500, 10**3, 2500, 5000], ['100', '250', '500', '1000', '2500', '5000'])
    plt.xlim(10**2, 5000)
    # 设置x轴为对数坐标
    plt.subplots_adjust(left=0.10, right=0.94, top=0.97, bottom=0.11)
    plt.show()
    plt.savefig(path)
    plt.close() 


def draw_heat(data, path):
    sns.set_theme(style='whitegrid')
    set_iclr_style(layout=3)
    names = list(data.columns)
    
    expanded = pd.DataFrame(data[names[1]].tolist(), index=data[names[0]]).T
    corr = expanded.corr()

    linear_colors = ['#F7FBFC', '#D6E6F2', '#B9D7EA', '#769FCD']
    # 定义自定义渐变色，从绿色 → 白色 → 红色
    
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom", linear_colors, N=256
    )

    plt.figure()
    ax = sns.heatmap(corr, annot=True, annot_kws={"color": 'black'}, cmap=custom_cmap, center=0.5, fmt=".2f", vmin=0, vmax=1)
    plt.xlabel('')
    plt.ylabel('')

# 在每个方格写数值
    # for i in range(corr.shape[0]):
    #     for j in range(corr.shape[1]):
    #         value = corr.iloc[i, j]
    #         ax.text(j+0.5, i+0.5, f"{value:.2f}",
    #                 ha="center", va="center", color="black")

    # cbar = ax.collections[0].colorbar
    # ax.tick_params(axis='both', which='major')
    # plt.ylim(0.2, 0.4)
    
    plt.subplots_adjust(left=0.06, right=1.05, top=0.97, bottom=0.09)
    # plt.show()
    plt.savefig(path)
    plt.close()

    
def draw_line(data, path):
    sns.set_theme(style='whitegrid')
    # set_iclr_style(layout=3)
    set_iclr_style(layout=2)
    names = list(data.columns)
    plt.figure()
    # figure_colors = ['#FF9494', '#FFCFB3', '#B7E0FF', '#4A628A', '#AA96DA', '#FCBAD3']
    figure_colors = ['#90BCD5', '#E76254', '#7976A2', '#4A5E65', '#E29957', '#86B5A1', '#B95A58', '#4292C6']
    colors = sns.color_palette(figure_colors)
        # 使用模型名称作为标签
    # plt.plot(subset[names[0]], subset[names[1]], label=model_name, linestyle=linestyle, marker=mark, color=color)
    sns.lineplot(x=names[0], y=names[1], data=data, hue=names[2], style=names[2], markers=True, palette=colors, dashes=False)
    # ax.legend(fontsize='24', loc='best', fancybox=True, framealpha=0.5)
    plt.legend(loc='upper left', fancybox=True, framealpha=0.5) # 图例
    plt.xlabel(names[0])
    plt.ylabel(names[1]) # Y轴标签

    # plt.legend(handles=legend_lines, loc='best')
    # ax.tick_params(axis='both', which='major', labelsize=24)
    # plt.ylim(0.2, 0.4)
    
    plt.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.15)
    plt.show()
    plt.savefig(path)
    plt.close()



def draw_dendrogram(data, path, method="average", n_clusters=None, max_d=None):
    sns.set_theme(style='whitegrid')
    set_iclr_style(layout=3)
    # ===== 1) 构造相关矩阵 =====
    names = list(data.columns)
    expanded = pd.DataFrame(data[names[1]].tolist(), index=data[names[0]]).T
    corr = expanded.corr()

    # ===== 2) 转换成距离矩阵 =====
    dist = (1 - corr)
    condensed = squareform(dist.values, checks=False)
    # linear_colors = [figure_colors[0], '#FF0000']
    custom_cmap = LinearSegmentedColormap.from_list("custom", figure_colors[:3], N=256)
    # ===== 3) 层次聚类 =====
    Z = linkage(condensed, method=method)

    # ===== 4) 绘制树状图 =====
    plt.figure()
    dendro = dendrogram(
        Z,
        labels=corr.columns,
        orientation="left",
        # leaf_font_size=14,
        color_threshold=None,
        above_threshold_color="black",
    )

    ax = plt.gca()
    heights = Z[:, 2]  # 距离值
    vmax = max(heights)  # 最大距离
    vmin = min(heights)  # 最小距离

    for i, line in enumerate(ax.collections):
        # 将每条边的距离值归一化，并映射到颜色范围
        color = custom_cmap((heights[i] - vmin) / (vmax - vmin))  # 归一化处理
        line.set_edgecolor(color)
        line.set_linewidth(2)
    

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_xlabel("Distance")
    ax.set_ylabel("")
    ax.tick_params(axis="y")
    ax.tick_params(axis="x")

    # ===== 5) 可选：画分割线 =====
    if n_clusters is not None:
        heights = Z[:, 2]
        k = len(heights) - (n_clusters - 1)
        if 0 <= k < len(heights):
            max_d = heights[k]
    if max_d is not None:
        ax.axvline(max_d, linestyle="--", color="red", linewidth=1.5)
        
    plt.subplots_adjust(left=0.03, right=0.93, top=0.97, bottom=0.07)
    # plt.show()
    # plt.tight_layout()
    plt.savefig(path)
    plt.close()
# 生成“更美观”的树状图（横向、加粗、网格、紧凑布局），并按3簇画参考线