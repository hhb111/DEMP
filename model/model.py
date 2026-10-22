import datetime
import math
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class GlobalGraphEncoder(Module):
    def __init__(self, opt, num_node, adj_all, num, shared_embedding_layer):
        super(GlobalGraphEncoder, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()

        # Aggregator

        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)
        self.embedding = shared_embedding_layer

    def sample(self, target, n_sample):
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def forward(self, inputs, mask_item, item):
        batch_size = item.shape[0]
        seqs_len = item.shape[1]
        item_neighbors = [item]
        weight_neighbors = []
        support_size = seqs_len
        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))
        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors
        session_info = []
        item_emb = self.embedding(inputs) * mask_item.float().unsqueeze(-1)
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))
        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        return h_global

class Hybrid_MoE(nn.Module):
    def __init__(self, opt, num_node, adj, num):
        super().__init__()
        self.dim = opt.hiddenSize
        self.k_coarse = 3 if getattr(opt, 'id_interests', None) is None else opt.id_interests
        self.k_img = 3 if getattr(opt, 'image_interests', None) is None else opt.image_interests
        self.k_txt = 3 if getattr(opt, 'text_interests', None) is None else opt.text_interests
        if min(self.k_coarse, self.k_img, self.k_txt) < 1:
            raise ValueError("Interest cardinalities must be positive.")
        self.interests = self.k_coarse + self.k_img + self.k_txt
        self.length = opt.length
        self.beta = opt.beta
        self.beta_coarse = opt.beta
        self.beta_img = opt.beta
        self.beta_txt= opt.beta
        self.gamma = opt.gamma
        self.lambda1 = 0.5 if opt.lambda1 is None else opt.lambda1
        self.lambda2 = opt.lambda2
        self.lambda3 = 1.0 if opt.lambda3 is None else opt.lambda3
        self.batch_size = opt.batch_size
        self.gate_variant = getattr(opt, 'gate_variant', 'seq_proto')
        valid_gate_variants = {'seq_proto', 'seq_no_proto', 'static', 'item'}
        if self.gate_variant not in valid_gate_variants:
            raise ValueError(f"Unknown gate_variant={self.gate_variant}. Expected one of {sorted(valid_gate_variants)}")
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        feature_dir = getattr(opt, 'dataset_dir', None)
        if feature_dir is None:
            feature_dir = os.path.join('.', 'datasets', opt.dataset)
        image_feature_data = np.load(os.path.join(feature_dir, 'image_feature.npy'))
        image_tensor = torch.tensor(image_feature_data, dtype=torch.float32)
        text_feature_data = np.load(os.path.join(feature_dir, 'text_feature.npy'))
        text_tensor = torch.tensor(text_feature_data, dtype=torch.float32)
        self.embedding_image = nn.Embedding.from_pretrained(image_tensor, freeze=True)
        self.embedding_text = nn.Embedding.from_pretrained(text_tensor, freeze=True)
        H_image = image_tensor.shape[1]
        H_text = text_tensor.shape[1]
        self.image_projection = nn.Linear(H_image, self.dim)
        self.text_projection = nn.Linear(H_text, self.dim)
        self.expert_global = GlobalGraphEncoder(opt, num_node, adj, num, self.embedding)

        self.w_1_id = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2_id = nn.Parameter(torch.Tensor(self.dim, self.k_coarse))
        self.glu1_id = nn.Linear(self.dim, self.k_coarse * self.dim)
        self.w_1_img = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2_img = nn.Parameter(torch.Tensor(self.dim, self.k_img))
        self.glu1_img = nn.Linear(self.dim, self.k_img * self.dim)

        self.w_1_txt = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2_txt = nn.Parameter(torch.Tensor(self.dim, self.k_txt))
        self.glu1_txt = nn.Linear(self.dim, self.k_txt * self.dim)
        self.num_envs =5

        self.scoring_gating_network = nn.Sequential(
            nn.Linear(self.dim+self.num_envs, self.dim), 
            nn.ReLU(),
            nn.Linear(self.dim, 3)
        )
        self.scoring_gating_network1 = nn.Sequential(
            nn.Linear(self.dim, self.dim), 
            nn.ReLU(),
            nn.Linear(self.dim, 3)
        )
        self.logits_norm = nn.LayerNorm(3)
        self.debias_alpha = nn.Parameter(torch.zeros(1, 1, 3))
        self.item_gating_network = nn.Linear(3, 3)

        self.register_buffer("env_prototypes", torch.randn(self.num_envs, self.dim))
        self.env_prototypes = F.normalize(self.env_prototypes, dim=-1)
        self.register_buffer("static_gate_weights", torch.full((3,), 1.0 / 3.0))

        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, weight in self.named_parameters():
            if "embedding_image.weight" in name or "embedding_text.weight" in name:
                continue
            weight.data.uniform_(-stdv, stdv)
        
    def update_prototypes(self,z_current):

        with torch.no_grad():

            z_norm = F.normalize(z_current, dim=-1)
            proto_norm = F.normalize(self.env_prototypes, dim=-1)

            sim = torch.matmul(z_norm, proto_norm.T)
            _, idx = torch.max(sim, dim=1)
            momentum = 0.99
            for i in range(self.num_envs):
                mask = (idx == i)
                if mask.sum() > 0:
                    new_center = z_norm[mask].mean(dim=0)
                    self.env_prototypes[i] = momentum * self.env_prototypes[i] + (1 - momentum) * new_center

    def causal_gating(self, final_interests, z_current, update_proto=True):

        if final_interests.dim() == 3:
            interest_input = torch.mean(final_interests, dim=0) # [B, D]
        else:
            interest_input = final_interests
    
        z_norm = F.normalize(z_current, dim=-1)           
        proto_norm = F.normalize(self.env_prototypes, dim=-1)
        similarity = torch.matmul(z_norm, proto_norm.T)   
        p_z_given_u = torch.softmax(similarity / 0.1, dim=-1)
        combined_features = torch.cat([interest_input, p_z_given_u], dim=-1) # [B, 2D]
        raw_logits = self.scoring_gating_network(combined_features) 
        final_weights = F.softmax(raw_logits, dim=-1) 
        if self.training and update_proto:
            self.update_prototypes(z_current)
            
        return final_weights

    def sequence_gating(self, final_interests, z_current, update_proto=True):
        if self.gate_variant == 'seq_proto':
            return self.causal_gating(final_interests, z_current, update_proto)

        if final_interests.dim() == 3:
            interest_input = torch.mean(final_interests, dim=0)
        else:
            interest_input = final_interests

        if self.gate_variant == 'seq_no_proto':
            raw_logits = self.scoring_gating_network1(interest_input)
            return F.softmax(raw_logits, dim=-1)

        if self.gate_variant == 'static':
            return self.static_gate_weights.unsqueeze(0).expand(interest_input.shape[0], -1)

        raise ValueError(f"sequence_gating does not support gate_variant={self.gate_variant}")

    def item_score_gating(self, score_stack):
        raw_logits = self.item_gating_network(score_stack)
        return F.softmax(raw_logits, dim=-1)
    
    def calculate_diversity_loss(self, beta, mask, k):
        if k <= 1: 
            return torch.tensor(0.0, device=beta.device)

        normalized_beta = F.normalize(beta, p=2, dim=1, eps=1e-12)
        sim_loss = torch.tensor(0.0, device=beta.device)
        
        for i in range(k):
            for j in range(i + 1, k):
                dot_product = torch.sum(normalized_beta[:, :, i] * normalized_beta[:, :, j], dim=1)
                sim_loss += torch.mean(torch.square(dot_product))
        
        num_pairs = k * (k - 1) / 2
        diversity_loss = sim_loss / num_pairs
        return diversity_loss

    def compute_scores(self, f_id_seq, f_image_seq, f_text_seq, mask,z_current,z_cf):
        batch_size = f_id_seq.shape[0]
        L = f_id_seq.shape[1]
        mask_expanded = mask.float().unsqueeze(-1)
        pos_emb = self.pos_embedding.weight[:L]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        nh_id = torch.matmul(torch.cat([pos_emb, f_id_seq], -1), self.w_1_id)
        nh_id = torch.tanh(nh_id)
        nh_id = torch.sigmoid(self.glu1_id(nh_id))
        nh_split_id = torch.split(nh_id, self.dim, dim=2)
        nh_id = torch.stack(nh_split_id, dim=3)
        w2_id = self.w_2_id.unsqueeze(0).repeat(nh_id.shape[0], nh_id.shape[1], 1, 1)
        beta_coarse = torch.sum(nh_id * w2_id, dim=2)

        nh_m = torch.matmul(torch.cat([pos_emb, f_image_seq], -1), self.w_1_img)
        nh_m = torch.tanh(nh_m)
        nh_m = torch.sigmoid(self.glu1_img(nh_m))
        nh_split_m = torch.split(nh_m, self.dim, dim=2)
        nh_m = torch.stack(nh_split_m, dim=3)
        w2_m = self.w_2_img.unsqueeze(0).repeat(nh_m.shape[0], nh_m.shape[1], 1, 1)
        beta_m = torch.sum(nh_m * w2_m, dim=2)

        nh_t = torch.matmul(torch.cat([pos_emb, f_text_seq], -1), self.w_1_txt)
        nh_t = torch.tanh(nh_t)
        nh_t = torch.sigmoid(self.glu1_txt(nh_t))
        nh_split_t = torch.split(nh_t, self.dim, dim=2)

        nh_t = torch.stack(nh_split_t, dim=3)
        w2_t = self.w_2_txt.unsqueeze(0).repeat(nh_t.shape[0], nh_t.shape[1], 1, 1)
        beta_t = torch.sum(nh_t * w2_t, dim=2)

        mask_expanded_coarse = mask_expanded.expand(-1, -1, self.k_coarse)
        mask_expanded_img = mask_expanded.expand(-1, -1, self.k_img)
        mask_expanded_txt = mask_expanded.expand(-1, -1, self.k_txt)
   
        beta_coarse = beta_coarse * mask_expanded_coarse
        beta_m = beta_m * mask_expanded_img
        beta_t = beta_t * mask_expanded_txt

        diversity_loss_coarse = self.calculate_diversity_loss(beta_coarse, mask.float(), self.k_coarse)
        diversity_loss_m = self.calculate_diversity_loss(beta_m, mask.float(), self.k_img)
        diversity_loss_t = self.calculate_diversity_loss(beta_t, mask.float(), self.k_txt)
        total_diversity_loss = diversity_loss_coarse + diversity_loss_m + diversity_loss_t

        beta_coarse_permuted = beta_coarse.permute(2, 0, 1).unsqueeze(-1)
        f_id_rep = f_id_seq.unsqueeze(0).repeat(self.k_coarse, 1, 1, 1)
        interests_coarse = torch.sum(beta_coarse_permuted * f_id_rep, dim=2)

        beta_m_permuted = beta_m.permute(2, 0, 1).unsqueeze(-1)
        f_image_rep = f_image_seq.unsqueeze(0).repeat(self.k_img, 1, 1, 1)
        interests_m = torch.sum(beta_m_permuted * f_image_rep, dim=2)

        beta_t_permuted = beta_t.permute(2, 0, 1).unsqueeze(-1)
        f_text_rep = f_text_seq.unsqueeze(0).repeat(self.k_txt, 1, 1, 1)
        interests_t = torch.sum(beta_t_permuted * f_text_rep, dim=2)

        final_interests = torch.cat([interests_coarse, interests_m, interests_t], dim=0)
        final_interests = F.normalize(final_interests, p=2.0, dim=-1)

        split_sections = [self.k_coarse, self.k_img, self.k_txt]
        interests_list = torch.split(final_interests, split_sections, dim=0)
        
        int_id_src = interests_list[0]   # [K_id, B, D]
        int_img_src = interests_list[1]  # [K_img, B, D]
        int_txt_src = interests_list[2]  # [K_txt, B, D]

        b_id = F.normalize(self.embedding.weight[1:], p=2.0, dim=-1)
        b_image = F.normalize(self.image_projection(self.embedding_image.weight[1:]), p=2.0, dim=-1)
        b_text = F.normalize(self.text_projection(self.embedding_text.weight[1:]), p=2.0, dim=-1)

        scores_id = torch.matmul(int_id_src, b_id.transpose(1, 0))*10.0
        scores_image = torch.matmul(int_img_src, b_image.transpose(1, 0))*10.0
        scores_text = torch.matmul(int_txt_src, b_text.transpose(1, 0))*10.0

        all_scores_stacked = torch.cat([scores_id, scores_image, scores_text], dim=0)

        if self.k_coarse > 1:
            s_id_pooled, _ = torch.max(scores_id, dim=0) # [B, N]
        else:
            s_id_pooled = scores_id.squeeze(0)

        if self.k_img > 1:
            s_img_pooled, _ = torch.max(scores_image, dim=0) # [B, N]
        else:
            s_img_pooled = scores_image.squeeze(0)

        if self.k_txt > 1:
            s_txt_pooled, _ = torch.max(scores_text, dim=0) # [B, N]
        else:
            s_txt_pooled = scores_text.squeeze(0)

        if self.gate_variant == 'item':
            score_stack = torch.stack([s_id_pooled, s_img_pooled, s_txt_pooled], dim=-1)
            item_weights = self.item_score_gating(score_stack)
            final_scores = torch.sum(item_weights * score_stack, dim=-1)
            scoring_weights = torch.mean(item_weights, dim=1)
            weights_cf = torch.mean(self.item_score_gating(score_stack.detach()), dim=1)
        else:
            scoring_weights = self.sequence_gating(final_interests, z_current, True) # [B, 3]
            weights_cf = self.sequence_gating(final_interests.detach(), z_cf.detach(), False)
            
            w_id = scoring_weights[:, 0].unsqueeze(-1)  # [B, 1]
            w_img = scoring_weights[:, 1].unsqueeze(-1) # [B, 1]
            w_txt = scoring_weights[:, 2].unsqueeze(-1) # [B, 1]

            final_scores = w_id * s_id_pooled + w_img * s_img_pooled + w_txt * s_txt_pooled
        final_scores_per_interest =final_scores
        #final_scores_per_interest =s_id_pooled +  s_img_pooled +  s_txt_pooled
        max_scores = final_scores_per_interest
        _, max_indices = torch.max(final_scores_per_interest, dim=1) 
        sum_scores = final_scores_per_interest
        scores_tuple = (max_scores, total_diversity_loss, sum_scores)
        weights_tuple = (max_indices.detach(), scoring_weights.detach(), final_interests.detach())
        
        return scores_tuple, weights_tuple, weights_cf

    def get_masked_mean(self, features, mask):
        mask_float = mask.float().unsqueeze(-1)
        sum_features = torch.sum(features * mask_float, dim=1)
        count = torch.sum(mask_float, dim=1) + 1e-9
        return sum_features / count

    def calculate_cross_modal_cl_loss(self, view1, view2,mask, temperature=0.2):
        mask = mask.bool()
        view1 = view1[mask]
        view2 =view2[mask]
        v1 = F.normalize(view1, p=2, dim=1)
        v2 = F.normalize(view2, p=2, dim=1)
        logits = torch.matmul(v1, v2.T) / temperature
        batch_size = v1.shape[0]
        labels = torch.arange(batch_size).to(v1.device)
        loss = self.loss_function(logits, labels)
        return loss

    def forward(self, inputs, adj, mask, items, alias_inputs):
        h_id = self.embedding(items)                
        h_id = F.normalize(h_id, p=2.0, dim=-1)
        h_image_raw = self.embedding_image(inputs)    
        h_image = self.image_projection(h_image_raw)  
        h_image = F.normalize(h_image, p=2.0, dim=-1)
        h_text_raw = self.embedding_text(inputs)      
        h_text = self.text_projection(h_text_raw)    
        h_text = F.normalize(h_text, p=2.0, dim=-1)

        f_image_uni = h_image
        f_text_uni = h_text                  
        f_global = self.expert_global(inputs, mask, items)                  
        get_id = lambda index: f_global[index][alias_inputs[index]]
        f_global = torch.stack([get_id(i) for i in torch.arange(len(alias_inputs)).long()])
        mask_float = mask.float().unsqueeze(-1)
        z_current = torch.sum(f_global * mask_float, dim=1) / (torch.sum(mask_float, dim=1) + 1e-9)
        mean_id = z_current
        mean_img = self.get_masked_mean(f_image_uni, mask)
        mean_txt = self.get_masked_mean(f_text_uni, mask)

        z_current = mean_id  + mean_img + mean_txt
        z_current = F.normalize(z_current, dim=-1) 
        
        z_cf = mean_img + mean_txt
        z_cf = F.normalize(z_cf, dim=-1)

        return f_global, f_image_uni, f_text_uni, z_current,z_cf,h_id,h_image,h_text

def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    f_id_seq, f_image_seq, f_text_seq, z_current, z_cf,h_id, h_image, h_text = model(inputs, adj, mask, items, alias_inputs)
    scores_tuple, weights_tuple,  weights_cf = model.compute_scores(f_id_seq, f_image_seq, f_text_seq, mask, z_current, z_cf)
    return targets, scores_tuple, weights_tuple, z_current,h_id,h_image,h_text, weights_cf

def train(model, train_data, valid_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0

    model_device = next(model.parameters()).device
    total_avg_scoring_weights = torch.zeros(3, device=model_device) 
    
    num_batches = 0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=True, pin_memory=True)

    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores_tuple, weights_tuple, z_current, h_id, h_image, h_text,weights_cf = forward(model, data)
        
        scores, diversity_loss, _ = scores_tuple
        max_indices, scoring_weights, final_interests_batch = weights_tuple
        
        targets = trans_to_cuda(targets).long()
        loss_main = model.loss_function(scores, targets - 1)
        if model.lambda2 is None:
            loss_div = torch.mean(diversity_loss * model.beta * model.beta)
        else:
            loss_div = torch.mean(diversity_loss * model.lambda2)
        
        mask = trans_to_cuda(data[3]).long()
        loss_cl_img = model.calculate_cross_modal_cl_loss(h_id, h_image, mask, temperature=0.1)
        loss_cl_txt = model.calculate_cross_modal_cl_loss(h_id, h_text, mask, temperature=0.1)
        
        total_cl_loss = model.lambda1 * (loss_cl_img + loss_cl_txt)
        
        target_cf = torch.tensor([0.0, model.gamma, 1.0 - model.gamma], device=weights_cf.device)
        target_cf = target_cf.unsqueeze(0).expand(weights_cf.shape[0], -1) 
        
        loss_cf = F.mse_loss(weights_cf, target_cf)
        
        #loss = loss_main + loss_div + total_cl_loss
        loss = loss_main + loss_div + total_cl_loss +  model.lambda3 * loss_cf
        loss.backward()
        model.optimizer.step()
        batch_avg_scoring = torch.mean(scoring_weights.detach(), dim=0)
        total_avg_scoring_weights += batch_avg_scoring
        num_batches += 1
        total_loss += loss.item()

    epoch_avg_scoring_weights = total_avg_scoring_weights / num_batches
    weights_np = epoch_avg_scoring_weights.cpu().numpy()


    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    test_result = test(model, valid_data)
    return test_result

def test(model, test_data, save_weights=False):
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit10, mrr10, ndcg10 = [], [], []
    hit20, mrr20, ndcg20 = [], [], []
    cov = set()
    
    all_weights = [] 
    all_env_ids = []
    with torch.no_grad():
        for data in test_loader:
            targets, scores_tuple, weights_tuple, z_current, h_id, h_image, h_text, weights_cf = forward(model, data)
            scores, diversity_loss, _ = scores_tuple
            
            _, scoring_weights, _ = weights_tuple
            
            if save_weights:
                all_weights.append(scoring_weights.cpu().numpy())                
                z_norm = F.normalize(z_current, dim=-1)
                proto_norm = F.normalize(model.env_prototypes, dim=-1)
                sim = torch.matmul(z_norm, proto_norm.T)
                _, env_indices = torch.max(sim, dim=1)
                all_env_ids.append(env_indices.cpu().numpy())

            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets = targets.numpy()
            _, _, _, mask, _, _ = data
            mask_np = mask.numpy()
            for score, target, m in zip(sub_scores, targets, mask_np):
                cov.update(score)
                target_item = target - 1
                score10 = score[:10]

                hit10.append(np.isin(target_item, score10))
                where10 = np.where(score10 == target_item)[0]
                if len(where10) == 0:
                    mrr10.append(0)
                    ndcg10.append(0)
                else:
                    rank10 = where10[0]
                    mrr10.append(1.0 / (rank10 + 1))
                    ndcg10.append(1.0 / np.log2(rank10 + 2))

                hit20.append(np.isin(target_item, score))
                where20 = np.where(score == target_item)[0]
                if len(where20) == 0:
                    mrr20.append(0)
                    ndcg20.append(0)
                else:
                    rank20 = where20[0]
                    mrr20.append(1.0 / (rank20 + 1))
                    ndcg20.append(1.0 / np.log2(rank20 + 2))

    if save_weights and len(all_weights) > 0:
        weight_matrix = np.concatenate(all_weights, axis=0) # [N_test, 3]
        np.save('gate_weights.npy', weight_matrix)
        print("【Success】Gate weights saved to 'gate_weights.npy'")
        env_id_array = np.concatenate(all_env_ids, axis=0)
        # 保存
        np.savez('env_analysis.npz', weights=weight_matrix, env_ids=env_id_array)
        print("Data saved to env_analysis.npz")

    result.append(np.mean(hit10) * 100)
    result.append(np.mean(mrr10) * 100)
    result.append(np.mean(ndcg10) * 100)
    result.append(np.mean(hit20) * 100)
    result.append(np.mean(mrr20) * 100)
    result.append(np.mean(ndcg20) * 100)
    result.append(len(cov))
    return result
