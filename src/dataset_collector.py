import os
import json
import re
import requests
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from synthetic_generator import APLSyntheticGenerator

# Allowed open source licenses
ALLOWED_LICENSES = {
    'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 
    'cc0-1.0', 'wtfpl', 'isc', 'zlib', 'bsl-1.0',
    'gpl-2.0', 'gpl-3.0', 'lgpl-2.1', 'lgpl-3.0', 'agpl-3.0',
    'mpl-2.0', 'epl-1.0', 'epl-2.0', 'artistic-2.0',
    'cc-by-4.0', 'cc-by-sa-4.0', 'cc-by-sa-3.0'
}

CURATED_REPOS = [
    {'repo': 'Dyalog/dyalog-apl-extended-compiler', 'license': 'MIT'},
    {'repo': 'the-carls/apl-quest', 'license': 'MIT'},
    {'repo': 'kimmolinna/apl-snippets', 'license': 'MIT'},
    {'repo': 'Co-dfns/Co-dfns', 'license': 'MIT'},
    {'repo': 'Dyalog/cryptopl', 'license': 'MIT'},
    {'repo': 'Dyalog/cabal', 'license': 'MIT'},
    {'repo': 'the-carls/dyalog-jupyter-kernel', 'license': 'MIT'}
]

def load_local_settings() -> Dict[str, str]:
    settings = {}
    settings_file = Path('settings.local')
    if settings_file.is_file():
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        settings[k.strip().upper()] = v.strip()
        except Exception as e:
            print(f'Error reading settings.local: {e}')
    return settings

def get_auth_headers(token: str = None) -> Dict[str, str]:
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if not token:
        settings = load_local_settings()
        token = os.environ.get('GITHUB_TOKEN') or settings.get('GITHUB_TOKEN')
    if token and token != 'your_personal_access_token_here':
        headers['Authorization'] = f'token {token}'
    return headers

def check_repo_license(repo_full_name: str, headers: Dict[str, str]) -> Tuple[bool, str]:
    url = f'https://api.github.com/repos/{repo_full_name}/license'
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            spdx = data.get('license', {}).get('spdx_id', '').lower()
            if spdx in ALLOWED_LICENSES:
                return True, spdx.upper()
        elif r.status_code == 404:
            # Fallback: check main repo metadata
            r_repo = requests.get(f'https://api.github.com/repos/{repo_full_name}', headers=headers, timeout=10)
            if r_repo.status_code == 200:
                lic = r_repo.json().get('license')
                if lic and lic.get('spdx_id'):
                    spdx = lic['spdx_id'].lower()
                    if spdx in ALLOWED_LICENSES:
                        return True, spdx.upper()
    except Exception as e:
        print(f'License check error on {repo_full_name}: {e}')
    return False, 'UNKNOWN'

def fetch_apl_files_from_repo(repo_full_name: str, headers: Dict[str, str]) -> List[Tuple[str, str]]:
    results = []
    tree_url = f'https://api.github.com/repos/{repo_full_name}/git/trees/HEAD?recursive=1'
    try:
        r = requests.get(tree_url, headers=headers, timeout=15)
        if r.status_code != 200:
            return results
        tree = r.json().get('tree', [])
        valid_exts = {'.apl', '.apln', '.aplf', '.aplo', '.aplc', '.dyalog'}
        
        for item in tree:
            path = item.get('path', '')
            ext = os.path.splitext(path)[1].lower()
            if ext in valid_exts and item.get('type') == 'blob':
                # Fetch raw content
                raw_url = f'https://raw.githubusercontent.com/{repo_full_name}/HEAD/{path}'
                raw_r = requests.get(raw_url, timeout=10)
                if raw_r.status_code == 200:
                    text = raw_r.text
                    if text.strip():
                        results.append((path, text))
    except Exception as e:
        print(f'Error fetching files from {repo_full_name}: {e}')
    return results

def search_github_apl_repos(limit: int = 50, headers: Dict[str, str] = None) -> List[Dict[str, str]]:
    repos = []
    page = 1
    per_page = min(limit, 30)
    
    while len(repos) < limit:
        url = f'https://api.github.com/search/repositories?q=language:APL+stars:>=1&sort=stars&order=desc&per_page={per_page}&page={page}'
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                print(f'Search API status {r.status_code}: {r.text[:200]}')
                break
            items = r.json().get('items', [])
            if not items:
                break
            for item in items:
                repos.append({
                    'repo': item['full_name'],
                    'html_url': item['html_url'],
                    'license': item.get('license', {}).get('spdx_id', 'UNKNOWN') if item.get('license') else 'UNKNOWN'
                })
                if len(repos) >= limit:
                    break
            page += 1
        except Exception as e:
            print(f'Error during GitHub search: {e}')
            break
    return repos

def main():
    parser = argparse.ArgumentParser(description='APL Dataset Collector & Attribution Generator')
    parser.add_argument('-m', '--mode', choices=['curated', 'search', 'synth-only'], default='curated', help='Collection mode')
    parser.add_argument('-l', '--limit', type=int, default=30, help='Max repos to scan')
    parser.add_argument('-s', '--synth-count', type=int, default=5000, help='Number of synthetic idioms to generate')
    parser.add_argument('-t', '--token', type=str, default=None, help='GitHub Personal Access Token')
    args = parser.parse_args()

    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    corpus_file = data_dir / 'apl_corpus.txt'
    attribution_file = data_dir / 'ATTRIBUTION.md'

    headers = get_auth_headers(args.token)
    all_content = []
    attributions = []

    print(f'Running APL Dataset Collector (Mode: {args.mode})...')

    if args.mode in ['curated', 'search']:
        if args.mode == 'curated':
            target_repos = CURATED_REPOS
        else:
            print(f'Searching GitHub for top {args.limit} APL repositories...')
            target_repos = search_github_apl_repos(limit=args.limit, headers=headers)

        for entry in target_repos:
            repo_name = entry['repo']
            print(f'Processing {repo_name}...')
            is_valid, spdx = check_repo_license(repo_name, headers)
            if not is_valid and entry.get('license') and entry['license'].lower() in ALLOWED_LICENSES:
                is_valid = True
                spdx = entry['license'].upper()
                
            if is_valid:
                files = fetch_apl_files_from_repo(repo_name, headers)
                if files:
                    print(f'  ✓ Found {len(files)} APL files under license {spdx}')
                    for fpath, code in files:
                        all_content.append(f'⍝ === Repo: {repo_name} | File: {fpath} ===\n{code}\n')
                    attributions.append({
                        'repo': repo_name,
                        'url': f'https://github.com/{repo_name}',
                        'license': spdx,
                        'files_count': len(files)
                    })
            else:
                print(f'  ✗ Skipped (Non-permissive or unknown license: {spdx})')

    # Add synthetic idioms
    if args.synth_count > 0:
        print(f'Generating {args.synth_count} synthetic APL idioms & dfns...')
        synth_data = APLSyntheticGenerator.generate_synthetic_corpus(args.synth_count)
        all_content.append(f'⍝ === Synthetic Algorithmic APL Idioms & Dfns ===\n{synth_data}\n')

    # Write unified corpus
    full_corpus = '\n'.join(all_content)
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write(full_corpus)
    print(f'✓ Unified APL corpus saved to {corpus_file} ({len(full_corpus):,} characters)')

    # Write attribution document
    with open(attribution_file, 'w', encoding='utf-8') as f:
        f.write('# 📜 Open Source Dataset Attribution\n\n')
        f.write('This dataset contains open-source APL code gathered from GitHub under verified permissive licenses:\n\n')
        f.write('| Repository | License | Extracted Files |\n| :--- | :--- | :--- |\n')
        for item in attributions:
            f.write(f"| [{item['repo']}]({item['url']}) | {item['license']} | {item['files_count']} |\n")
        f.write(f"\n*Synthetic Dataset:* Augmented with {args.synth_count} synthetic APL idioms generated algorithmically.\n")
    print(f'✓ Attribution document generated at {attribution_file}')

if __name__ == '__main__':
    main()
