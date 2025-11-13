import re
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def add_n_to_content(content):
    
    c_list = content.split(' ')
    conter = 0
    out = ''
    for con in c_list:
        conter += 1
        out += con 
        if conter == 2:
            out += '\n'
            conter = 0
        else:
            out += ' '
    return out

def extract_mermaid_graph(mermaid_code):
    # 正则表达式模式
    node_pattern = re.compile(r'(\w+)(?:\(\(([^)]*)\)\)|\{([^}]*)\}|\[([^\]]*)\])')  # 匹配各类节点
    edge_pattern = re.compile(r'(\w+)\s*--\s*(.*?)\s*-->\s*(\w+)')                    # 带标签的边
    simple_edge_pattern = re.compile(r'(\w+)\s*-->\s*(\w+)')                          # 无标签的边

    nodes = {'A': 'Start'}
    edges = []

    lines = mermaid_code.split('\n') 

    for line in lines:
        line_new = line.strip()
        line_new = line_new.replace('<br>', ' ').replace('\"', '').replace('"', '').replace("\\", '')
        
        print(line_new)
        if not line_new or line_new.startswith('%%') or line_new.startswith('classDef'):
            continue  # 忽略注释和样式
        
        # 提取节点定义（独立行中的节点）
        node_match = node_pattern.match(line_new)
        if node_match:
            node_id = node_match.group(1)
            content = node_match.group(2) or node_match.group(3) or node_match.group(4) or ''
            content = add_n_to_content(content)
            nodes[node_id] = content.strip()
    
            continue
        
        # 提取带标签的边和节点定义（行内节点）
        edge_match = edge_pattern.search(line_new)
        if edge_match:
            src, label, dest = edge_match.groups()
            label = add_n_to_content(label)
            edges.append({'source': src, 'target': dest, 'label': label.strip()})
            
            # 提取行内源节点定义
            src_def = node_pattern.search(line_new.split('--')[0])
            if src_def:
                src_id = src_def.group(1)
                content = src_def.group(2) or src_def.group(3) or src_def.group(4) or ''
                content = add_n_to_content(content)
                nodes[src_id] = content.strip()
        
                
            
            # 提取行内目标节点定义
            dest_def = node_pattern.search(line_new.split('-->')[-1])
            if dest_def:
                dest_id = dest_def.group(1)
                content = dest_def.group(2) or dest_def.group(3) or dest_def.group(4) or ''
                content = add_n_to_content(content)
                nodes[dest_id] = content.strip()
        
                
            continue
        
        # 提取无标签的边
        simple_edge_match = simple_edge_pattern.search(line_new)
        if simple_edge_match:
            src, dest = simple_edge_match.groups()
            edges.append({'source': src, 'target': dest, 'label': ''})
            
            # 提取行内节点定义
            src_def = node_pattern.search(line_new.split('-->')[0])
            if src_def:
                src_id = src_def.group(1)
                content = src_def.group(2) or src_def.group(3) or src_def.group(4) or ''
                content = add_n_to_content(content)
                nodes[src_id] = content.strip()
                # print(nodes)

            
            dest_def = node_pattern.search(line_new.split('-->')[-1])
            if dest_def:
                dest_id = dest_def.group(1)
                content = dest_def.group(2) or dest_def.group(3) or dest_def.group(4) or ''
                content = add_n_to_content(content)
                nodes[dest_id] = content.strip()
                # print(nodes)
                
        


    # 转换为标准输出格式
    nodes_list = [{'id': k, 'content': v} for k, v in nodes.items()]
    return {'nodes': nodes_list, 'edges': edges}


def mermaid_to_networkx(mermaid_code):
    # 使用之前定义的提取函数
    graph_data = extract_mermaid_graph(mermaid_code)
    print(graph_data['nodes'])
    print(graph_data['edges'])
    
    G = nx.DiGraph()
    
    # 添加带属性的节点
    for node in graph_data['nodes']:
        G.add_node(node['id'], label=node['content'])
    
    # 添加带标签的边
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], label=edge['label'])
    
    return G

def _hierarchy_pos(G, root, width=1., vert_gap=0.4, vert_loc=0, xcenter=0.5):
    pos = {root: (xcenter, vert_loc)}
    neighbors = list(G.successors(root))
    
    if len(neighbors) != 0:
        dx = width/len(neighbors)
        nextx = xcenter - width/2 - dx/2
        
        for neighbor in neighbors:
            nextx += dx
            pos.update(_hierarchy_pos(
                G, neighbor, 
                width=dx, 
                vert_gap=vert_gap,
                vert_loc=vert_loc-vert_gap, 
                xcenter=nextx))
    
    # 水平对齐优化
    min_x = min(x for x, y in pos.values())
    max_x = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = ( (pos[node][0] - min_x) / (max_x - min_x), 
                     pos[node][1] )
    
    return pos

def draw_tree(G):
    plt.figure(figsize=(12, 8))
    
    # 自动识别根节点（没有父节点的节点）
    roots = [node for node in G.nodes if G.in_degree(node) == 0]
    root = roots[0] if roots else list(G.nodes)[0]
    
    # 使用树状布局算法
    pos = _hierarchy_pos(G, root)
    
    # 绘制节点
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(
        G, pos, 
        node_shape="s",  # 方框节点
        node_color="lightblue",
        edgecolors="darkblue",
        node_size=2500
    )
    
    # 特殊样式：为根节点添加装饰
    nx.draw_networkx_nodes(
        G, pos, nodelist=[root],
        node_shape="D",  # 菱形节点
        node_color="gold",
        node_size=3000
    )
    
    # 绘制节点标签（带自动换行）
    for node, (x, y) in pos.items():
        plt.text(x, y, 
                 "\n".join([node_labels[node][i:i+15] for i in range(0, len(node_labels[node]), 15)]),
                 ha='center', 
                 va='center',
                 fontsize=9,
                 bbox=dict(facecolor='white', 
                         edgecolor='darkblue',
                         boxstyle='round,pad=0.3'))

    # 绘制边（带流动动画效果样式）
    edge_labels = nx.get_edge_attributes(G, 'label')
    for (u, v), label in edge_labels.items():
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color="gray",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
            connectionstyle=f"arc3,rad={0.3 if label == 'Yes' else -0.3}"  # 不同标签不同弯曲方向
        )
        
        # 动态计算标签位置
        mid_point = [(pos[u][0] + pos[v][0])/2, (pos[u][1] + pos[v][1])/2]
        plt.text(mid_point[0], mid_point[1], 
                label,
                color='darkred',
                fontsize=8,
                ha='center',
                va='center',
                backgroundcolor='white')

    plt.title("Decision Tree Visualization", pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def draw_graph(G):
    
    # 使用 pydot 的 graphviz_layout 布局
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")  # "dot" 用于树形布局
        # pos = nx.spring_layout(G, k=0.1, iterations=100) 
    except ImportError:
        raise ImportError("需要安装 pydot 和 Graphviz，运行 `pip install pydot` 并确保 Graphviz 已安装。")
    
    # 绘制节点
    node_labels = nx.get_node_attributes(G, 'label')  # 获取节点的标签
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    # 绘制边
    edge_labels = nx.get_edge_attributes(G, 'label')  # 获取边的标签
    nx.draw_networkx_edges(
        G, pos, 
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20,
        connectionstyle='arc3,rad=0.05'  # 带弧度的连线
    )
    
    # 显示边标签
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='darkred',
        label_pos=0.5,  # 标签位置调整
        rotate=False
    )
    
    plt.title("Mermaid Flowchart Visualization")
    plt.axis('off')
    
    # 自动调整边避免重叠
    plt.tight_layout()



data_path = 'data/medqa/logic/train4.jsonl'
with open(data_path, 'r') as f:
    lines = f.readlines()


graph_list = []

for i in range(len(lines)):

    datas = lines[i].split('```')[1]
    try:
        datas = datas.split('mermaid')[1]
    except:
        pass
    datas = datas.replace('\\n', '\n')
    datas = datas.replace('((Start))', '')
    
    print(datas)
    G = mermaid_to_networkx(datas)
    plt.figure(figsize=(15, 10))
    draw_graph(G)
    plt.savefig(f"logic_output/mermaid_{i}.png")
    
    graph_list.append(G)
    
# ged = gm.GraphEditDistance(1, 1, 1, 1) 
# result = ged.compare([graph_list[0], graph_list[1]], None)



# reindexed_graph_list = [nx.convert_node_labels_to_integers(g) for g in graph_list]
# from karateclub import Graph2Vec

# # 使用 Graph2Vec 进行图嵌入
# model = Graph2Vec(dimensions=64)
# model.fit(reindexed_graph_list)
# embeddings = model.get_embedding()

# np.save('logic_output/graph_embeddings.npy', embeddings)

# 计算图的余弦相似度
# import pdb; pdb.set_trace()
# similarity = cosine_similarity(embeddings, embeddings)

# print(similarity)


# umap_emb = umap.UMAP(n_neighbors=5, metric='cosine').fit_transform(embeddings)
# plt.figure(figsize=(10,10))
# plt.scatter(umap_emb[:, 0], umap_emb[:, 1], s=3)
# plt.tight_layout()
# plt.savefig("umap_emb.png", dpi=300)