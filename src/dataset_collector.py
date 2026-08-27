"""
Dataset Collector & License Attribution Generator for APL SLM.
Scrapes permissible open-source APL repositories from GitHub and GitLab and blends them
with algorithmically generated synthetic idioms.
"""

import os
import sys
import json
import time
import hashlib
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set

from synthetic_generator import APLSyntheticGenerator

# Allowed open-source licenses (SPDX IDs)
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

VALID_APL_EXTENSIONS = {'.apl', '.apln', '.aplf', '.aplo', '.aplc', '.dyalog', '.mip'}


def load_local_settings() -> Dict[str, str]:
    """Loads key-value pairs from settings.local if present."""
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
            print(f'Warning: Error reading settings.local: {e}')
    return settings


def get_auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Constructs GitHub API authentication headers."""
    headers = {'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'APL-LLM-Collector/1.0'}
    if not token:
        settings = load_local_settings()
        token = os.environ.get('GITHUB_TOKEN') or settings.get('GITHUB_TOKEN')
    if token and token != 'your_personal_access_token_here':
        headers['Authorization'] = f'token {token}'
    return headers


def check_repo_license(repo_full_name: str, headers: Dict[str, str]) -> Tuple[bool, str]:
    """Verifies whether a GitHub repository has a permissible open-source license."""
    url = f'https://api.github.com/repos/{repo_full_name}/license'
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            spdx = data.get('license', {}).get('spdx_id', '').lower()
            if spdx in ALLOWED_LICENSES:
                return True, spdx.upper()
        elif r.status_code == 404:
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


def fetch_apl_files_from_repo(repo_full_name: str, headers: Dict[str, str], seen_hashes: Optional[Set[str]] = None) -> List[Tuple[str, str]]:
    """Recursively fetches text content of all APL code blobs in a GitHub repository."""
    if seen_hashes is None:
        seen_hashes = set()
    results = []
    tree_url = f'https://api.github.com/repos/{repo_full_name}/git/trees/HEAD?recursive=1'
    try:
        r = requests.get(tree_url, headers=headers, timeout=15)
        if r.status_code != 200:
            return results
        tree = r.json().get('tree', [])

        for item in tree:
            path = item.get('path', '')
            ext = os.path.splitext(path)[1].lower()
            if ext in VALID_APL_EXTENSIONS and item.get('type') == 'blob':
                raw_url = f'https://raw.githubusercontent.com/{repo_full_name}/HEAD/{path}'
                raw_r = requests.get(raw_url, timeout=10)
                if raw_r.status_code == 200 and raw_r.text.strip():
                    cleaned = raw_r.text.strip()
                    code_hash = hashlib.sha256(cleaned.encode('utf-8')).hexdigest()
                    if code_hash not in seen_hashes:
                        seen_hashes.add(code_hash)
                        results.append((path, cleaned))
    except Exception as e:
        print(f'Error fetching files from {repo_full_name}: {e}')
    return results


def search_github_apl_repos(limit: int = 50, headers: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """Searches GitHub for top-starred APL repositories."""
    repos = []
    page = 1
    per_page = min(limit, 30) if limit > 0 else 30
    max_scan = limit if limit > 0 else 100

    while len(repos) < max_scan:
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
                if len(repos) >= max_scan:
                    break
            page += 1
        except Exception as e:
            print(f'Error during GitHub search: {e}')
            break
    return repos


def fetch_gitlab_apl_projects(seen_hashes: Optional[Set[str]] = None, limit: int = 30) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Scrapes public APL repositories from GitLab API."""
    if seen_hashes is None:
        seen_hashes = set()
    code_list = []
    attributions = []
    headers = {'User-Agent': 'APL-LLM-Collector/1.0'}
    print('Fetching public APL repositories from GitLab...')

    max_projects_to_scan = limit if (limit and limit > 0) else 40
    page = 1
    scanned_count = 0

    while scanned_count < max_projects_to_scan:
        per_page = min(50, max_projects_to_scan - scanned_count)
        url = f'https://gitlab.com/api/v4/projects?search=apl&visibility=public&per_page={per_page}&page={page}'
        try:
            res = None
            for attempt in range(2):
                try:
                    res = requests.get(url, headers=headers, timeout=8)
                    if res.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    if attempt == 1:
                        raise
                    time.sleep(1)

            if res and res.status_code == 200:
                projects = res.json()
                if not projects:
                    break

                for p in projects:
                    scanned_count += 1
                    pname = p.get('path_with_namespace')
                    web_url = p.get('web_url')
                    pid = p.get('id')
                    default_branch = p.get('default_branch') or 'main'

                    try:
                        tree_url = f'https://gitlab.com/api/v4/projects/{pid}/repository/tree?recursive=true&per_page=40'
                        tres = requests.get(tree_url, headers=headers, timeout=6)
                        if tres.status_code == 200:
                            tree = tres.json()
                            apl_files = [f for f in tree if os.path.splitext(f.get('name', ''))[1].lower() in VALID_APL_EXTENSIONS]
                            repo_files_count = 0
                            for fentry in apl_files[:10]:
                                fpath = fentry.get('path')
                                raw_url = f'https://gitlab.com/api/v4/projects/{pid}/repository/files/{requests.utils.quote(fpath, safe="")}/raw?ref={default_branch}'
                                try:
                                    raw_res = requests.get(raw_url, headers=headers, timeout=6)
                                    if raw_res.status_code != 200 and default_branch != 'master':
                                        raw_url = f'https://gitlab.com/api/v4/projects/{pid}/repository/files/{requests.utils.quote(fpath, safe="")}/raw?ref=master'
                                        raw_res = requests.get(raw_url, headers=headers, timeout=6)
                                    if raw_res.status_code == 200:
                                        content = raw_res.text.strip()
                                        if len(content) > 10:
                                            chash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                                            if chash not in seen_hashes:
                                                seen_hashes.add(chash)
                                                code_list.append(f'⍝ === Platform: GitLab | Repo: {pname} | File: {fpath} ===\n{content}\n')
                                                repo_files_count += 1
                                                print(f'  [+] Downloaded from GitLab: {pname}/{fpath} ({len(content)} chars)')
                                except requests.exceptions.RequestException:
                                    continue
                            if repo_files_count > 0:
                                attributions.append({
                                    'platform': 'GitLab',
                                    'repo': pname,
                                    'url': web_url,
                                    'license': 'Open Source',
                                    'files_count': repo_files_count
                                })
                    except requests.exceptions.RequestException:
                        continue

                    if limit and limit > 0 and len(code_list) >= limit:
                        break

                if len(projects) < per_page or (limit and limit > 0 and len(code_list) >= limit):
                    break
                page += 1
            else:
                break
        except Exception as e:
            print(f'  [-] Skipped remaining GitLab search: {e}')
            break

    print(f'  [+] Collected {len(code_list)} files from {scanned_count} GitLab projects.')
    return code_list, attributions


def build_dataset(
    mode: str = 'curated',
    limit: int = 30,
    synth_count: int = 5000,
    token: Optional[str] = None,
    sources: Optional[List[str]] = None
):
    """Builds APL corpus dataset from selected sources."""
    if sources is None:
        sources = ['github', 'gitlab', 'synthetic']

    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    corpus_file = data_dir / 'apl_corpus.txt'
    attribution_file = data_dir / 'ATTRIBUTION.md'

    headers = get_auth_headers(token)
    seen_hashes: Set[str] = set()
    all_content: List[str] = []
    attributions: List[Dict[str, Any]] = []

    print(f'Running APL Dataset Collector (Mode: {mode}, Sources: {", ".join(sources)})...')

    if 'github' in sources and mode in ['curated', 'search', 'all']:
        if mode == 'curated':
            target_repos = CURATED_REPOS
        else:
            print(f'Searching GitHub for top {limit} APL repositories...')
            target_repos = search_github_apl_repos(limit=limit, headers=headers)

        for entry in target_repos:
            repo_name = entry['repo']
            print(f'Processing GitHub: {repo_name}...')
            is_valid, spdx = check_repo_license(repo_name, headers)
            if not is_valid and entry.get('license') and entry['license'].lower() in ALLOWED_LICENSES:
                is_valid = True
                spdx = entry['license'].upper()
            elif mode == 'all':
                is_valid = True
                spdx = spdx if spdx != 'UNKNOWN' else 'Open Source'

            if is_valid:
                files = fetch_apl_files_from_repo(repo_name, headers, seen_hashes=seen_hashes)
                if files:
                    print(f'  ✓ Found {len(files)} APL files under license {spdx}')
                    for fpath, code in files:
                        all_content.append(f'⍝ === Platform: GitHub | Repo: {repo_name} | File: {fpath} ===\n{code}\n')
                    attributions.append({
                        'platform': 'GitHub',
                        'repo': repo_name,
                        'url': f'https://github.com/{repo_name}',
                        'license': spdx,
                        'files_count': len(files)
                    })
            else:
                print(f'  ✗ Skipped (Non-permissive or unknown license: {spdx})')

    if 'gitlab' in sources and mode in ['curated', 'search', 'all']:
        gitlab_code, gitlab_attrs = fetch_gitlab_apl_projects(seen_hashes=seen_hashes, limit=limit)
        all_content.extend(gitlab_code)
        attributions.extend(gitlab_attrs)

    # Add synthetic idioms
    if 'synthetic' in sources and synth_count > 0:
        print(f'Generating {synth_count:,} synthetic APL idioms & dfns...')
        synth_data = APLSyntheticGenerator.generate_synthetic_corpus(synth_count)
        all_content.append(f'⍝ === Synthetic Algorithmic APL Idioms & Dfns ===\n{synth_data}\n')

    # Write unified corpus
    full_corpus = '\n'.join(all_content)
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write(full_corpus)
    print(f'✓ Unified APL corpus saved to {corpus_file} ({len(full_corpus):,} characters, {len(all_content)} blocks)')

    # Write attribution document
    with open(attribution_file, 'w', encoding='utf-8') as f:
        f.write('# 📜 Open Source Dataset Attribution\n\n')
        f.write('This dataset contains open-source APL code gathered from GitHub and GitLab under verified open-source and permissive licenses:\n\n')
        f.write('| Platform | Repository | License | Extracted Files | Link |\n| :--- | :--- | :--- | :--- | :--- |\n')
        for item in attributions:
            f.write(f"| {item['platform']} | `{item['repo']}` | {item['license']} | {item['files_count']} | [Link]({item['url']}) |\n")
        if 'synthetic' in sources and synth_count > 0:
            f.write(f"\n*Synthetic Dataset:* Augmented with {synth_count:,} synthetic APL idioms generated algorithmically.\n")
    print(f'✓ Attribution document generated at {attribution_file}')


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description='APL Dataset Collector & Attribution Generator')
    parser.add_argument('-m', '--mode', choices=['curated', 'search', 'all', 'synth-only'], default='curated', help='Collection mode')
    parser.add_argument('-l', '--limit', type=int, default=30, help='Max repos to scan')
    parser.add_argument('-s', '--synth-count', type=int, default=5000, help='Number of synthetic idioms to generate')
    parser.add_argument('-t', '--token', type=str, default=None, help='GitHub Personal Access Token')
    parser.add_argument(
        '--sources',
        nargs='+',
        default=['github', 'gitlab', 'synthetic'],
        choices=['github', 'gitlab', 'synthetic'],
        help='Data sources to include during collection.'
    )
    args = parser.parse_args()

    actual_sources = args.sources
    if args.mode == 'synth-only':
        actual_sources = ['synthetic']

    build_dataset(
        mode=args.mode,
        limit=args.limit,
        synth_count=args.synth_count,
        token=args.token,
        sources=actual_sources
    )


if __name__ == '__main__':
    main()

