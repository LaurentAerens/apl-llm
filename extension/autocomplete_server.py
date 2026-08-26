import sys
import json
import argparse
from pathlib import Path
from autocomplete import APLCompleter

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stdin.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description='APL Autocomplete Daemon Server')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    args = parser.parse_args()

    try:
        completer = APLCompleter(args.checkpoint)
        sys.stderr.write(f'APL Completer loaded successfully from {args.checkpoint}\n')
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f'Failed to load model: {e}\n')
        sys.stderr.flush()
        sys.exit(1)

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            req_id = req.get('id', 0)
            prompt = req.get('prompt', '')
            max_tokens = req.get('max_tokens', 64)
            temp = req.get('temp', 0.2)
            top_k = req.get('top_k', 3)

            completion = completer.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                top_k=top_k
            )

            res = {
                'id': req_id,
                'completion': completion
            }
            sys.stdout.write(json.dumps(res, ensure_ascii=False) + '\n')
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f'Error processing request: {e}\n')
            sys.stderr.flush()

if __name__ == '__main__':
    main()
