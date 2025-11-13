import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

path = 'logic_output/graph_embeddings.npy'

embeddings = np.load(path)
import umap
import matplotlib.pyplot as plt
umap_emb = umap.UMAP(n_neighbors=5, random_state=0).fit_transform(embeddings)

km = KMeans(n_clusters=40, random_state=0).fit(umap_emb)
km_pr = km.predict(umap_emb)

index_list = [i for i in range(len(km_pr))]
index_list = np.array(index_list)
index_list_0 = index_list[km_pr == 0]
print(index_list_0)
print('------------------')
index_list_0 = index_list[km_pr == 8]
print(index_list_0)


d3_color_list = ["#4269d0","#efb118","#ff725c","#6cc5b0","#3ca951","#ff8ab7","#a463f2","#97bbf5","#9c6b4e","#9498a0"]
d3_color_list += ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
d3_color_list += ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#fb9a99","#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#ffff99","#b15928"]
d3_color_list += d3_color_list
d3_color_list += d3_color_list

colors = d3_color_list[:40]
cmap = ListedColormap(colors)


plt.figure(figsize=(10,10))

scatter = plt.scatter(
    umap_emb[:, 0],
    umap_emb[:, 1],
    s=3,
    c=km_pr,  # 直接使用簇编号
    cmap=cmap  # 使用自定义颜色映射
)

legend = plt.legend(
    *scatter.legend_elements(),
    title="Clusters",
    bbox_to_anchor=(1.05, 1),  # 将图例移到右侧
    loc='upper left'
)

for i, center in enumerate(km.cluster_centers_):
    plt.text(
        center[0], 
        center[1],
        s=str(i),  # 这里可以替换成实际类别名称
        fontsize=9,
        ha='center',
        va='center',
        color=colors[i],  # 使用与簇一致的颜色
        weight='bold',
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",  # 白色背景
            edgecolor=colors[i],  # 边框颜色与簇一致
            lw=1,
            alpha=0.8
        )
    )


# plt.savefig('logic_output/graph_umap.png')
plt.show()
