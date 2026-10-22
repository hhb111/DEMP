import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import *

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData, train_len=None):    #inputData is a 训练的二重列表[[1, 2, 3], [4, 5, 6]]
    len_data = [len(nowData) for nowData in inputData]  #len_data 是每一个训练序列的长度[3,3]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence and pad with 0 to the max length 
    # eg：max_len=5 us_pois = [[3, 2, 1, 0, 0], [6, 5, 4, 0, 0]]
    # mask is used to indicate the valid positions in the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


def handle_adj(adj_dict, n_entity, sample_num, num_dict=None): #邻接节点列表，item数量，采样数量，紧接节点权重
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            # 邻居数 >= 需要的数量：不放回抽样
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
             # 邻居数 < 需要的数量：有放回抽样（允许重复）
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])
    #adj_entity shape: (n_entity, sample_num) 邻接节点矩阵
    #num_entity shape: (n_entity, sample_num) 邻接节点权重矩阵
    return adj_entity, num_entity


class Data(Dataset):
    def __init__(self, data, train_len=None):
        inputs, mask, max_len = handle_data(data[0], train_len)
        #[[3, 2, 1, 0, 0], [6, 5, 4, 0, 0]]  inputs
        #[[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]]  mask
        # 5
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        #[4,7]  
        self.mask = np.asarray(mask)
        self.length = len(data[0])  #有多少个训练序列
        self.max_len = max_len      #最长的训练序列长度

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]

        max_n_node = self.max_len
        node = np.unique(u_input)            #去重，得到所有的节点
        items = node.tolist() + (max_n_node - len(node)) * [0]           
        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1       #自环边
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1         #自环边
            if adj[v][u] == 2:        #如果u和v之间已经有一条边了，则建立双向边
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2           #u->v建立一条out边
                adj[v][u] = 3           #v->u建立一条in边

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input)]

    def __len__(self):
        return self.length

def visualize_three_modalities(model, num_samples=2000):
    """
    model: 训练好的 Hybrid_MoE 模型实例
    num_samples: 随机采样的物品数量 (全部画太慢且乱，建议采样)
    """
    model.eval()
    
    # 1. 获取物品总数 (假设 embedding 第0位是 padding，从1开始或者直接随机采)
    num_items = model.embedding.weight.shape[0]
    
    # 随机选择索引 (排除 0 号 padding 如果需要)
    indices = np.random.choice(range(1, num_items), size=min(num_samples, num_items-1), replace=False)
    indices_tensor = torch.LongTensor(indices).cuda() if torch.cuda.is_available() else torch.LongTensor(indices)

    with torch.no_grad():
        # --- A. 获取 ID Embedding ---
        # 对应 forward 里的 h_id = self.embedding(items)
        emb_id = model.embedding(indices_tensor)
        emb_id = torch.nn.functional.normalize(emb_id, p=2.0, dim=-1) # 保持和 forward 一致的归一化

        # --- B. 获取 Image Embedding ---
        # 对应 forward 里的 h_image = self.image_projection(self.embedding_image(inputs))
        # 注意：这里直接取 embedding_image 的权重，不需要 inputs
        raw_img = model.embedding_image(indices_tensor)
        emb_img = model.image_projection(raw_img)
        emb_img = torch.nn.functional.normalize(emb_img, p=2.0, dim=-1)

        # --- C. 获取 Text Embedding ---
        # 对应 forward 里的 h_text = self.text_projection(self.embedding_text(inputs))
        raw_txt = model.embedding_text(indices_tensor)
        emb_txt = model.text_projection(raw_txt)
        emb_txt = torch.nn.functional.normalize(emb_txt, p=2.0, dim=-1)

    # 转为 Numpy
    data_id = emb_id.cpu().numpy()
    data_img = emb_img.cpu().numpy()
    data_txt = emb_txt.cpu().numpy()

    print(f"Sampling {len(indices)} items for visualization...")
    print(f"Shapes - ID: {data_id.shape}, IMG: {data_img.shape}, TXT: {data_txt.shape}")

    # 2. t-SNE 降维
    # 必须把三个模态拼在一起做 t-SNE，才能在同一个坐标系下比较
    combined_data = np.vstack([data_id, data_img, data_txt])
    
    print("Running t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    combined_2d = tsne.fit_transform(combined_data)

    # 拆分回三个部分
    len_data = len(data_id)
    tsne_id = combined_2d[:len_data]
    tsne_img = combined_2d[len_data:2*len_data]
    tsne_txt = combined_2d[2*len_data:]

    # 3. 绘图
    plot_stylish_distribution(tsne_id, tsne_img, tsne_txt)


def plot_stylish_distribution(id_coords, img_coords, txt_coords):
    # 1. 设置画板和风格
    fig, ax = plt.subplots(figsize=(7, 6)) # 长宽比例调整
    
    # 开启网格线 (这是右图显精致的关键)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.2, zorder=0)

    # 定义通用的描边样式 (关键代码)
    common_style = {
        'edgecolors': 'white',  # 【核心】白色描边
        'linewidth': 1.0,       # 描边宽度
        'alpha': 0.9,           # 不透明度 (高一点看起来更实)
        'zorder': 10            # 保证点画在网格线上面
    }

    # 2. 绘制散点
    # ID: 蓝色圆点 (Circle)
    ax.scatter(id_coords[:, 0], id_coords[:, 1], 
               c='#4c8ec6', marker='o', s=80, label='ID', 
               **common_style)
    
    # Image: 绿色倒三角 (或者你在用的叉号) - 这里模仿右图用了三角形
    ax.scatter(img_coords[:, 0], img_coords[:, 1], 
               c='#5ab885', marker='^', s=90, label='Image', 
               **common_style)
    
    # Text: 橙色正方形 (或圆形)
    ax.scatter(txt_coords[:, 0], txt_coords[:, 1], 
               c='#e69f47', marker='s', s=70, label='Text', 
               **common_style)

    # 3. 坐标轴美化 (模仿右图)
    # 隐藏上方和右方的边框线 (Spines)，或者保留做成方框
    # 右图是一个完整的方框，所以我们保留默认，但把刻度朝内
    ax.tick_params(direction='in', length=4, width=1, colors='black', grid_alpha=0.5)

    # 设置图例
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), 
              ncol=3, frameon=False, fontsize=12, handletextpad=0.1)
    
    plt.title("Arts Dataset Distribution", y=1.1, fontsize=14, fontfamily='serif')
    plt.tight_layout()
    plt.savefig("multimodal_distribution.pdf", bbox_inches='tight') # 保存为矢量图
    plt.show()

def analyze_macro_disentanglement(model, test_loader, device='cuda'):
    print("开始进行宏观解耦分析 (Macro-Level Disentanglement Analysis)...")
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_loader, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    all_similarities = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Collecting Embeddings"):
            # 1. 运行模型 Forward
            # 注意：根据你最新的 forward 返回值修改解包
            # 假设 forward 返回: targets, scores_tuple, weights_tuple, loss_cluster, z_current, mean_img, mean_txt, ...
            # 只需要 z_current 和 mean_img
            ret = forward(model, data)
            z_current = ret[3]  # 假设 z_current 是第5个返回
            mean_img = ret[4]   # 假设 mean_img 是第6个返回
            
            # 2. 计算余弦相似度 (Cosine Similarity)
            # dim=1 表示在特征维度计算
            # sim 形状: [Batch_Size]
            sim = F.cosine_similarity(z_current, mean_img, dim=1)
            
            # 3. 收集结果
            all_similarities.append(sim.cpu().numpy())
            
    # 拼接所有 Batch
    all_similarities = np.concatenate(all_similarities)
    
    # 保存数据（可选）
    np.save('session_similarity.npy', all_similarities)
    
    return all_similarities

def plot_similarity_histogram(similarities, dataset_name="Sports"):
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    # bins=50: 把数据分成50个区间
    # alpha=0.7: 透明度
    # density=True: 显示概率密度而不是绝对数量
    n, bins, patches = plt.hist(similarities, bins=50, alpha=0.7, color='#1f77b4', 
                                edgecolor='black', density=True, label='Ours (with Orthogonality)')
    
    # 添加一条 x=0 的虚线，作为参考线
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Orthogonality (0)')
    
    # 装饰图表
    plt.title(f'Distribution of Cosine Similarity between ID & Visual Context\n({dataset_name} Dataset)', fontsize=16)
    plt.xlabel('Cosine Similarity', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 设置 x 轴范围 (-1 到 1)
    plt.xlim(-1, 1)
    
    # 保存
    save_path = f'macro_disentanglement_{dataset_name}.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"图表已保存至: {save_path}")

    # 打印统计信息
    print(f"平均相似度: {np.mean(similarities):.4f} (越接近0越好)")
    print(f"相似度标准差: {np.std(similarities):.4f}")