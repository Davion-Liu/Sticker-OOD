import argparse
import os
import torch
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--joint_epochs', type=int, default=10)
    parser.add_argument('--align_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    

    os.makedirs(args.save_dir, exist_ok=True)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XAlignSR().to(device)
    

    print("Starting Warmup Training...")
    warmup_loader = get_dataloader('data/warmup.json', args.batch_size, 'warmup')
    warmup_trainer = WarmupTrainer(model)
    
    for epoch in range(args.warmup_epochs):
        total_loss = 0
        pbar = tqdm(warmup_loader, desc=f'Warmup Epoch {epoch+1}')
        for batch in pbar:
            loss = warmup_trainer.warmup_step(batch)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f'Epoch {epoch+1} Avg Loss: {total_loss/len(warmup_loader):.4f}')
    
    torch.save(model.state_dict(), f'{args.save_dir}/warmup.pth')
    

    print("\nStarting Joint Training...")
    train_loader = get_dataloader('data/train.json', args.batch_size, 'train')
    joint_trainer = JointTrainer(model)
    
    for epoch in range(args.joint_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Joint Epoch {epoch+1}')
        for batch in pbar:
            loss = joint_trainer.joint_step(batch)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f'Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}')
    
    torch.save(model.state_dict(), f'{args.save_dir}/joint.pth')
    
    print("\nStarting Alignment Training...")
    align_trainer = AlignmentTrainer(model, gamma=0.5)
    
    for epoch in range(args.align_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Align Epoch {epoch+1}')
        for batch in pbar:
            loss = align_trainer.align_step(batch)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f'Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}')
    
    torch.save(model.state_dict(), f'{args.save_dir}/final.pth')
    print("Training Completed!")

if __name__ == '__main__':
    main()
