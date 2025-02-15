import torch
import torch.nn as nn
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

class TextIntentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").text_model
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]

class ImageIntentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").vision_model
    
    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state[:,0,:]

class CrossModalAlignment(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, t_emb, v_emb):
        Q = self.query_proj(t_emb)
        K = self.key_proj(v_emb)
        V = self.value_proj(v_emb)
        attn = torch.softmax((Q @ K.transpose(-2,-1)) * self.scale, dim=-1)
        return attn @ V

class XAlignSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextIntentEncoder()
        self.image_encoder = ImageIntentEncoder()
        self.alignment = CrossModalAlignment(512)
    
    def text_contrastive_loss(self, q_emb, t_pos, t_neg):
        pos_logits = torch.sum(q_emb * t_pos, dim=-1)
        neg_logits = q_emb @ t_neg.T
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(q_emb.device)
        return nn.CrossEntropyLoss()(logits, labels)
    
    def image_contrastive_loss(self, v_emb, v_pos, v_neg):
        pos_logits = torch.sum(v_emb * v_pos, dim=-1)
        neg_logits = v_emb @ v_neg.T
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(v_emb.device)
        return nn.CrossEntropyLoss()(logits, labels)
    
    def alignment_loss(self, q_emb, h_s, neg_emb):
        pos_sim = torch.cosine_similarity(q_emb, h_s)
        neg_sim = q_emb @ neg_emb.T
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(q_emb.device)
        return nn.CrossEntropyLoss()(logits, labels)
    
    def forward(self, q_input, t_pos_input, v_pos_input, t_neg_input, v_neg_input):
        q_emb = self.text_encoder(**q_input)
        t_pos_emb = self.text_encoder(**t_pos_input)
        t_neg_emb = self.text_encoder(**t_neg_input)
        v_pos_emb = self.image_encoder(v_pos_input)
        v_neg_emb = self.image_encoder(v_neg_input)
        
        L_text = self.text_contrastive_loss(q_emb, t_pos_emb, t_neg_emb)
        L_img = self.image_contrastive_loss(v_pos_emb, v_pos_emb, v_neg_emb)
        
        t_s = self.text_encoder(**t_pos_input)
        v_s = self.image_encoder(v_pos_input)
        h_s = self.alignment(t_s, v_s)
        L_expr = torch.norm(t_s - h_s, p=2, dim=-1).mean()
        
        neg_emb = self.alignment(t_neg_emb, v_neg_emb)
        L_align = self.alignment_loss(q_emb, h_s, neg_emb)
        
        return L_text, L_img, L_expr, L_align

class WarmupTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters())
    
    def warmup_step(self, batch):
        text_emb = self.model.text_encoder(batch['input_ids'], batch['attention_mask'])
        image_emb = self.model.image_encoder(batch['pixel_values'])
        logits = text_emb @ image_emb.T
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

class JointTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters())
    
    def joint_step(self, batch):
        L_text, L_img, _, _ = self.model(**batch)
        alpha = L_text / (L_text + L_img)
        loss = alpha * L_text + (1 - alpha) * L_img
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

class AlignmentTrainer:
    def __init__(self, model, gamma=0.5):
        self.model = model
        self.gamma = gamma
        self.optimizer = torch.optim.AdamW(model.parameters())
    
    def align_step(self, batch):
        L_text, L_img, L_expr, L_align = self.model(**batch)
        loss = self.gamma * L_expr + L_align
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
