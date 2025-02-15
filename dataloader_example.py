import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ChineseCLIPProcessor

class StickerDataset(Dataset):
    def __init__(self, data_path, mode='warmup'):
        with open(data_path) as f:
            self.data = json.load(f)
        self.processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        self.mode = mode
        self.text_fields = ['description', 'ocr_text', 'ip_tags'] 
        
    def __len__(self):
        return len(self.data)
    
    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.mode == 'warmup':
           
            selected_field = self.text_fields[torch.randint(0, len(self.text_fields), (1,)).item()]
            return {
                'text': item['metadata'][selected_field],
                'image_path': item['image_path']
            }
        else:
            
            main_data = {
                'query': item['enhanced_query']['combined'],
                'pos_sticker': {
                    'text_variants': item['pos_sticker']['text_variants'],
                    'image_path': item['pos_sticker']['image_path']
                }
            }
            
            
            neg_indices = torch.randperm(len(self.data))[:5]
            main_data['neg_stickers'] = [{
                'text': self.data[i]['metadata']['ocr_text'],
                'image_path': self.data[i]['image_path']
            } for i in neg_indices if i != idx]
            
            return main_data

class DataCollator:
    def __init__(self, mode='warmup'):
        self.mode = mode
        self.processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        
    def warmup_collate(self, batch):
        texts = [item['text'] for item in batch]
        images = [self.process_image(item['image_path']) for item in batch]
        
        text_inputs = self.processor(text=texts, padding=True, return_tensors="pt")
        image_inputs = torch.stack(images)
        
        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'pixel_values': image_inputs
        }
    
    def train_collate(self, batch):
       
        pos_texts = [item['pos_sticker']['text_variants'] for item in batch]
        pos_images = [self.process_image(item['pos_sticker']['image_path']) for item in batch]
        
       
        neg_texts = [neg['text'] for item in batch for neg in item['neg_stickers']]
        neg_images = [self.process_image(neg['image_path']) for item in batch for neg in item['neg_stickers']]
        
       
        queries = [item['query'] for item in batch]
        query_inputs = self.processor(text=queries, padding=True, return_tensors="pt")
        
    
        selected_pos_texts = [variants[torch.randint(0, len(variants), (1,))] for variants in pos_texts]
        pos_text_inputs = self.processor(text=selected_pos_texts, padding=True, return_tensors="pt")
        neg_text_inputs = self.processor(text=neg_texts, padding=True, return_tensors="pt")
        
        return {
            'q_input': {
                'input_ids': query_inputs['input_ids'],
                'attention_mask': query_inputs['attention_mask']
            },
            't_pos_input': {
                'input_ids': pos_text_inputs['input_ids'],
                'attention_mask': pos_text_inputs['attention_mask']
            },
            'v_pos_input': torch.stack(pos_images),
            't_neg_input': {
                'input_ids': neg_text_inputs['input_ids'],
                'attention_mask': neg_text_inputs['attention_mask']
            },
            'v_neg_input': torch.stack(neg_images)
        }

def get_dataloader(data_path, batch_size=32, mode='warmup'):
    dataset = StickerDataset(data_path, mode)
    collator = DataCollator(mode)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator.warmup_collate if mode == 'warmup' else collator.train_collate,
        num_workers=4,
        pin_memory=True
    )


warmup_loader = get_dataloader('path/to/warmup_data.json', mode='warmup')
train_loader = get_dataloader('path/to/train_data.json', mode='train')
