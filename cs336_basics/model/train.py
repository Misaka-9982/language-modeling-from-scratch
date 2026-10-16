# Training Loop

import torch
import argparse
import numpy as np

from modules import *
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--rms_eps", type=float, default=1e-5)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")  # 保存目录
    parser.add_argument("--resume_checkpoint", type=str, default=None)  # 加载目录
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)

    return parser.parse_args()
    

def main(args):
    # 确保目录存在
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 初始化model和optimizer
    model = TransformerLM(args.d_model, args.num_heads, args.d_ff, args.rope_theta,
                          args.vocab_size, args.context_length, args.num_layers, args.rms_eps,
                          args.device, dtype=torch.float32)
    optimizer = AdamW(model.state_dict(), args.lr, (args.beta1, args.beta2), args.eps, args.weight_decay)

    # 加载数据集
    # 用memmap的方式加载tokenized数据到虚拟内存，避免一次性加载整个数据集到物理内存
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r')

    # 出于性能考虑暂时注释掉了
    # 显式检查一下数据集的ids中是否包括超过vocab_size的值
    # assert max(train_data) < args.vocab_size
    # assert max(val_data) < args.vocab_size

    start_step = 0
    # 从checkpoint恢复训练
    if args.resume_checkpoint is not None:
        resume_checkpoint_path = Path(args.resume_checkpoint)
        start_step = load_checkpoint(src=resume_checkpoint_path, model=model, optimizer=optimizer)

    # 训练循环
    for step in range(start_step, args.max_steps):
        print(f'step: {step + 1}')

        # 取训练batch
        train_input, train_target = dataloader(dataset=train_data, batch_size=args.batch_size, context_length=args.context_length, device=args.device)

        y = model(train_input)

        loss = cross_entropy_loss(y, train_target)
        print(f'train loss: {loss.item()}')

        # 清除历史梯度
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # 计算验证集loss
        if (step + 1) % args.eval_interval == 0:
            # 如果增加了dropout层，eval模式会影响模型行为
            model.eval()

            with torch.no_grad():
                val_input, val_target = dataloader(dataset=val_data, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
                val_y = model(val_input)
                val_loss = cross_entropy_loss(val_y, val_target)
                print(f'val loss: {val_loss.item()}')

            model.train()

        # 保存checkpoint
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, iteration=step + 1, 
                            out=checkpoint_dir / f'checkpoint_step_{step + 1}.pt')

    # final checkpoint
    print('===训练结束===')
    save_checkpoint(model, optimizer, iteration=args.max_steps, 
                    out=checkpoint_dir / f'checkpoint_step_{args.max_steps}.pt')
    print(f'Final checkpoint已保存到: {checkpoint_dir / f'checkpoint_step_{args.max_steps}.pt'}')



if __name__ == '__main__':
    args = parse_args()
    main(args)
# load函数  输出loss图 