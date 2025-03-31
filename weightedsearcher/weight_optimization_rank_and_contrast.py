import torch
import torch.nn.functional as F
import logging

class DistributionForUpdate:
    def __init__(self, ranker_list, weight_vec, doc_list, lr, max_iterations, top_num=10, bottom_num=10, alpha=0.5, mask=10, min_weight=0.01):
        self.ranker_list = ranker_list
        self.weight_vec = {k: torch.nn.Parameter(torch.tensor([v], dtype=torch.float32)) for k, v in weight_vec.items()}
        self.doc_list = doc_list
        self.lr = lr
        self.max_iterations = max_iterations
        self.top_num = top_num  # 超参数，前后文档数量
        self.min_weight = min_weight  # 最小权重值
        self.bottom_num = bottom_num 
        self.alpha = alpha
        self.mask = mask
        self.optimizer = torch.optim.Adam(self.weight_vec.values(), self.lr)
        self.reorder_doc_list()
        self.contrast_loss_log = []
        self.rank_loss_log = []
        self.initial_all_doc_scores = None
        self.tau = None
        
    def reorder_doc_list(self):
        id_to_doc = {doc['id']: doc for doc in self.doc_list}
        self.ordered_doc_list = [id_to_doc[ranker['id']] for ranker in self.ranker_list if ranker['id'] in id_to_doc]
    
    def compute_doc_scores(self, use_rank=True, doc_indices=None):
        scores = []

        if doc_indices is None:
            doc_list = self.ordered_doc_list if use_rank else self.doc_list
        else:
            doc_list = [self.ordered_doc_list[idx] for idx in doc_indices] if use_rank else [self.doc_list[idx] for idx in doc_indices] 
        
        for doc in doc_list:
            score = sum(doc['token_score'][token] * self.weight_vec[token] for token in doc['token_score'] if token in self.weight_vec)
            scores.append(score)
        return torch.stack(scores)

    def scale_weights(self, scale_factor):
        for param in self.weight_vec.values():
            param.data = (param.data * scale_factor + 1) / 2
            # param.data = param.data * scale_factor
    
    def contrast_loss(self, top_scores, bottom_scores):
        if self.tau is None:
            self.tau = (top_scores.median() - bottom_scores.median()).item()
        
        loss = 0.0
        num_elements = self.top_num * self.top_num
        for top_score in top_scores:
            for bottom_score in bottom_scores:
                loss += torch.max(torch.tensor(0.0), 1 - (top_score - bottom_score) / self.tau)
    
        # if top_scores.numel() > 1:
        #     variance = top_scores.var()
        # else:
        #     variance = torch.tensor(0.0)        
        # variance_penalty = self.lambda_var * variance
        # loss += variance_penalty  
              
        if num_elements == 0:
            return torch.tensor(0.0)
        else:
            return loss / num_elements
    
    def log_contrast_loss(self, top_scores, bottom_scores):
        loss = 0.0
        num_elements = self.top_num * self.top_num
        for top_score in top_scores:
            for bottom_score in bottom_scores:
                total_score = top_score + bottom_score
                top_score = top_score / total_score
                bottom_score = bottom_score / total_score
                max_score = torch.max(top_score, bottom_score)
                numerator = torch.exp(top_score - max_score)
                denominator = torch.exp(top_score - max_score) + torch.exp(bottom_score - max_score)
                loss += -torch.log(numerator / denominator)
                
        if top_scores.numel() > 1:
            variance = top_scores.var()
        else:
            variance = torch.tensor(0.0)        
        variance_penalty = self.lambda_var * variance
        loss += variance_penalty
        
        if num_elements == 0:
            return torch.tensor(0.0)
        else:
            return loss / num_elements
    
    def softmax(self, scores):
        return F.softmax(scores, dim=0)
    
    def pairwise_cross_entropy_loss(self, input_dist, target_dist, use_rank=True):
        n = len(target_dist)
        if use_rank:
            input_diff = input_dist.unsqueeze(1) - input_dist.unsqueeze(0)
            mask = torch.triu(torch.ones(n, n), diagonal=1)
            mask[self.mask:] = 0
            masked_diff = input_diff[mask.bool()]
        else:
            input_diff = input_dist.unsqueeze(1) - input_dist.unsqueeze(0)
            mask = torch.triu(torch.ones(n, n), diagonal=1)
            masked_diff = input_diff[mask.bool()]
                        
        target_diff = mask[mask.bool()]
        loss = F.binary_cross_entropy_with_logits(masked_diff, target_diff, reduction='sum')
        
        if mask.sum() == 0:
            return torch.tensor(0.0)
        else:
            return loss /  mask.sum()
        
    import torch

    def pairwise_cross_entropy_loss_gpu(self, input_dist, target_dist, use_rank=True, device='cuda:1'):
        # 将 input_dist 和 target_dist 移到 GPU
        input_dist = input_dist.to(device)
        target_dist = target_dist.to(device)
        
        n = len(target_dist)
        
        if use_rank:
            # 计算输入分布的差异
            input_diff = input_dist.unsqueeze(1) - input_dist.unsqueeze(0)
            
            # 创建掩码，保持上三角部分
            mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)
            
            # 如果 self.mask 存在
            mask[self.mask:] = 0
            masked_diff = input_diff[mask.bool()]
        else:
            input_diff = input_dist.unsqueeze(1) - input_dist.unsqueeze(0)
            mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)
            masked_diff = input_diff[mask.bool()]
        
        target_diff = mask[mask.bool()]
        
        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(masked_diff, target_diff, reduction='sum')
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        else:
            return loss / mask.sum()
    

    
    def update_weights(self, use_rank = False):
        if self.initial_all_doc_scores is None:
            self.initial_all_doc_scores = self.compute_doc_scores(use_rank).squeeze()
        
        initial_top_score = self.initial_all_doc_scores[list(range(self.top_num))].detach()

        self.optimizer.zero_grad()
        top_indices = list(range(self.top_num))
        bottom_indices = list(range(len(self.doc_list) - self.bottom_num, len(self.doc_list)))        
        top_scores = self.compute_doc_scores(use_rank, top_indices)
        bottom_scores = self.compute_doc_scores(use_rank, bottom_indices)
        contrast_loss = self.contrast_loss(top_scores, bottom_scores)

        if use_rank:
            self.dist_A_scores = torch.tensor([doc['ranker_score'] for doc in self.ranker_list], dtype=torch.float32)
            self.dist_B_score = self.compute_doc_scores(use_rank)
            self.dist_B_score = torch.squeeze(self.dist_B_score)
            rank_loss = self.pairwise_cross_entropy_loss(self.dist_B_score, self.dist_A_scores)
        else:
            top_scores = torch.squeeze(top_scores)
            rank_loss = self.pairwise_cross_entropy_loss(top_scores, initial_top_score, use_rank)
        
        total_loss = rank_loss + contrast_loss
        total_loss.backward()
        self.optimizer.step()
        for param in self.weight_vec.values():
            param.data = torch.clamp(param.data, min=self.min_weight)
               
        return rank_loss, contrast_loss
    
    def iterate_process(self, delta_threshold=0.0001, use_rank=True):
        previous_total_loss = float('inf')
        for i in range(self.max_iterations):
            rank_loss, contrast_loss = self.update_weights(use_rank)
            total_loss = self.alpha * rank_loss + (1 - self.alpha) * contrast_loss
            if i % 20 == 0:
                logger_text = f"LR: {self.lr} Iteration {i+1}: rank_loss = {rank_loss.item()}, contrast_loss = {contrast_loss.item()}, total_loss = {total_loss.item()}"
                self.rank_loss_log.append(logger_text)
            total_loss_delta = abs(total_loss.item() - previous_total_loss)
            if total_loss_delta < delta_threshold:
                logger_text = f"LR: {self.lr} Iteration {i+1}: rank_loss = {rank_loss.item()}, contrast_loss = {contrast_loss.item()}, total_loss = {total_loss.item()}"
                self.rank_loss_log.append(logger_text)
                break        
            previous_total_loss = total_loss.item()
        final_B_score_sum = self.compute_doc_scores().sum().item()
        
        initial_B_score_sum = self.initial_all_doc_scores.sum().item()
        scale_factor = initial_B_score_sum / final_B_score_sum
        self.scale_weights(scale_factor)

        logger_text = f"initial_B_score_sum: {initial_B_score_sum} and final_B_score_sum: {final_B_score_sum}, so the scale_factor is {scale_factor}"
        self.rank_loss_log.append(logger_text)
    
    def save_updated_weights(self):
        updated_weights = {token: param.item() for token, param in self.weight_vec.items()}
        logger_text = f"updated_weights: {updated_weights}"
        self.rank_loss_log.append(logger_text)
        return updated_weights, self.rank_loss_log