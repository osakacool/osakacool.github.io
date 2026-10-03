import geopandas as gpd
import contextily as cx
from shapely.geometry import Polygon
import matplotlib.patches as mpatches

# ==========================================
# Figure 1: 研究区位置图 (Location Map)
# ==========================================
def plot_study_area_map():
    # 模拟西安大唐不夜城边界 (Bounding Box)
    minx, miny, maxx, maxy = 108.945, 34.215, 108.965, 34.235
    bbox = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    gdf = gpd.GeoDataFrame({'name': ['Grand Tang Everbright City']}, geometry=[bbox], crs="EPSG:4326")
    gdf_proj = gdf.to_crs(epsg=3857) # 转换为Web Mercator以匹配底图
    
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf_proj.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=3, label='Study Area')
    
    # 添加开源底图 (OSM)
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom=14)
    
    ax.set_title('Figure 1: Location of Grand Tang Everbright City, Xi\'an', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    ax.axis('off') # 隐藏坐标轴以符合地图美学
    plt.tight_layout()
    plt.savefig('Fig1_Study_Area_Map.png', dpi=600, bbox_inches='tight')
    plt.show()

# ==========================================
# Figure 5: 夜间旅游功能区空间分布图 (Spatial Distribution)
# ==========================================
def plot_spatial_classification():
    # 模拟 100x100 网格的分类结果 (体现论文中的空间层级：线性核心+树枝状餐饮)
    np.random.seed(42)
    grid_size = 100
    classes = np.zeros((grid_size, grid_size), dtype=int)
    
    # 1. Core Consumption (线性走廊, 38.3%)
    classes[40:60, 20:80] = 1 
    # 2. Dining Cluster (树枝状分支, 27.9%)
    for i in range(20, 80, 10):
        classes[20:40, i:i+3] = 2
        classes[60:80, i:i+3] = 2
    # 3. Cultural Performance (聚集节点, 15.8%)
    classes[45:55, 45:55] = 3
    # 4. Transport Hub (边缘节点, 18.0%)
    classes[10:20, 10:20] = 4
    classes[80:90, 80:90] = 4
    # 5. Non-Tourism (背景, 剩余)
    classes[classes == 0] = 5 
    
    # 自定义学术色带 (Colorblind friendly)
    cmap = plt.cm.colors.ListedColormap(['#2b83ba', '#abdda4', '#fdae61', '#d7191c', '#f0f0f0'])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(classes, cmap=cmap, norm=norm, interpolation='nearest')
    
    # 添加图例
    labels = ['Core Consumption (38.3%)', 'Dining Cluster (27.9%)', 'Cultural Performance (15.8%)', 
              'Transport Hub (18.0%)', 'Non-Tourism']
    patches = [mpatches.Patch(color=cmap(i), label=labels[i]) for i in range(5)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    ax.set_title('Figure 5: Spatial Distribution of Classified Nocturnal Tourism Zones', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加比例尺和指北针 (示意)
    ax.text(0.05, 0.05, '500m', transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.85),
                arrowprops=dict(facecolor='black', width=5, headwidth=10), 
                transform=ax.transAxes, fontsize=14, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Fig5_Spatial_Distribution.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_study_area_map()
    plot_spatial_classification()