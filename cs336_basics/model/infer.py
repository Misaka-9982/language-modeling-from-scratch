# Generate text

import torch
import argparse
import numpy as np

from cs336_basics.model.modules import *
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_generated_tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_p", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda")

    # model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)  # d_ff=1344大概是d_model=512的8/3倍，且对齐到64的倍数
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--rms_eps", type=float, default=1e-5)
    parser.add_argument("--resume_checkpoint", type=str, required=True)



    return parser.parse_args()


def main(args):
    model = model = TransformerLM(args.d_model, args.num_heads, args.d_ff, args.rope_theta,
                              args.vocab_size, args.context_length, args.num_layers, args.rms_eps,
                              args.device, dtype=torch.float32)
    tokenizer = Tokenizer.from_files(vocab_filepath='cs336_basics/BPE_Tokenizer/tokenizer_vocab_10k_TinyStoriesV2-GPT4-train.pkl', 
                                         merges_filepath='cs336_basics/BPE_Tokenizer/tokenizer_merges_10k_TinyStoriesV2-GPT4-train.pkl', 
                                         special_tokens=['<|endoftext|>'])
    
    resume_checkpoint_path = Path(args.resume_checkpoint)
    in_dict = torch.load(resume_checkpoint_path)
    model.load_state_dict(in_dict['model'])
    generated_text = decoding(model=model, tokenizer=tokenizer, prompt=args.prompt, 
                              max_generated_tokens=args.max_generated_tokens, 
                              temperature=args.temperature, top_p=args.top_p, device=args.device)
    print(f'\nOutput Text: {generated_text}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
