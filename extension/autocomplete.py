import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizer import APLTokenizer
from model import APL_SLM

class APLCompleter:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = APLTokenizer()

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        saved_args = ckpt.get('args', {})

        self.model = APL_SLM(
            vocab_size=ckpt.get('vocab_size', self.tokenizer.vocab_size),
            n_layer=saved_args.get('n_layer', 4),
            n_head=saved_args.get('n_head', 4),
            n_embd=saved_args.get('n_embd', 64),
            max_seq_len=saved_args.get('seq_len', 1024)
        ).to(self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 64, 
        temperature: float = 0.7, 
        top_k: int = 5,
        stop_on_balanced_newline: bool = True
    ) -> str:
        prompt_tokens = self.tokenizer.encode(prompt)
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.bos_id]

        depth_seq = self.tokenizer.compute_depth_sequences(prompt_tokens)
        current_depth = depth_seq[-1] if depth_seq else 0

        x_tok = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        x_depth = torch.tensor([depth_seq], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _, kv_caches = self.model(x_tok, x_depth, use_cache=True)

        next_token_logits = logits[:, -1, :]
        generated_tokens = []

        for _ in range(max_tokens):
            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token in (self.tokenizer.eos_id, self.tokenizer.pad_id):
                break

            generated_tokens.append(next_token)
            
            # Update structural depth
            info = self.tokenizer.get_token_info(next_token)
            current_depth = max(0, current_depth + info.paren_delta + info.bracket_delta + info.dfn_delta)

            # Stopping condition: reached end of expression or balanced newline
            if stop_on_balanced_newline and current_depth == 0 and info.text in ('\n', '\r'):
                break

            # Step KV cache single token projection
            next_tok_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            next_depth_tensor = torch.tensor([[current_depth]], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits, _, kv_caches = self.model(
                    next_tok_tensor, 
                    next_depth_tensor, 
                    kv_caches=kv_caches, 
                    use_cache=True
                )
            next_token_logits = logits[:, -1, :]

        return self.tokenizer.decode(generated_tokens)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='APL Autocomplete CLI')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/baseline/apl_slm_best.pt', help='Model path')
    parser.add_argument('--prompt', type=str, default=None, help='APL code prefix to autocomplete')
    parser.add_argument('--max_tokens', type=int, default=64, help='Max tokens')
    parser.add_argument('--temp', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=5, help='Top K')
    args = parser.parse_args()

    if not Path(args.checkpoint).is_file():
        print(f'Error: Checkpoint {args.checkpoint} not found. Please train the model first with src/train.py')
        return

    completer = APLCompleter(args.checkpoint)

    if args.prompt:
        comp = completer.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k)
        print(f'Prompt:     {args.prompt}')
        print(f'Completion: {comp}')
    else:
        print('Interactive APL Autocomplete CLI (Type :q to quit)')
        while True:
            try:
                prompt = input('APL> ')
                if prompt.strip() == ':q':
                    break
                comp = completer.generate(prompt, max_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k)
                print(f'{prompt}{comp}')
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == '__main__':
    main()
