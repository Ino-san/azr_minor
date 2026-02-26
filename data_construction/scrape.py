import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque

class RacketGuideScraper:
    def __init__(self, start_url, output_file="racket_guide_dataset.jsonl", max_pages=50):
        self.start_url = start_url
        self.output_file = output_file
        self.max_pages = max_pages
        self.visited = set()
        self.queue = deque([start_url])
        
        parsed_start = urlparse(start_url)
        self.base_domain = parsed_start.netloc
        
        # Guide以下のみに制限
        self.target_prefix = "/guide/"

    def fetch_page(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; RacketGuideScraper/1.0)'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                self.remove_noise(soup)
                return soup
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None

    def remove_noise(self, soup):
        # メニューバーなどの不要要素を削除
        noise_selectors = ['.tocview', '.nav-content', '.bottom-nav', '.refpar', 'header', 'footer']
        for selector in noise_selectors:
            for tag in soup.select(selector):
                tag.decompose()

    def extract_pure_definitions(self, raw_text):
        """コードブロックからREPLプロンプトと実行結果を除去"""
        lines = raw_text.split('\n')
        kept_lines = []
        state = 'REPL_TRASH' 

        for line in lines:
            line = line.replace('\xa0', ' ').rstrip()
            if not line: continue

            if re.match(r'^\s*>', line):
                state = 'REPL_TRASH'
                continue

            if re.match(r'^\s+', line):
                if state == 'CODE': kept_lines.append(line)
                continue
            else:
                if line.startswith('(') or line.startswith('['):
                    state = 'CODE'
                    kept_lines.append(line)
                elif line.startswith('#lang') or line.startswith('#:'):
                    state = 'CODE'
                    kept_lines.append(line)
                else:
                    state = 'REPL_TRASH'

        return "\n".join(kept_lines)

    def extract_pairs(self, soup, source_url):
        extracted_data = []
        
        # 1. 候補をすべて取得
        # .Scribble-Racket と blockquote table の両方を取得し、重複があれば統合
        candidates = soup.select('.Scribble-Racket')
        if not candidates:
            candidates = soup.select('blockquote table')
            
        # リスト内の重複オブジェクトを除去（念のため）
        candidates = list({id(x): x for x in candidates}.values())

        # 2. 【追加】親子関係の解消（最も内側のブロックだけを残す）
        # 「自分の中に、リスト内の別の候補が含まれている」場合、自分は親（外枠）なので除外する
        code_blocks = []
        for block in candidates:
            is_parent = False
            for other in candidates:
                if block is other: 
                    continue
                # もし other が block の子孫(descendants)にあるなら、blockは親
                if other in block.descendants:
                    is_parent = True
                    break
            
            # 親でなければ（＝一番内側なら）採用
            if not is_parent:
                code_blocks.append(block)

        # デバッグ: フィルタリング前後の数を比較したい場合
        # print(f"  -> Debug: {len(candidates)} candidates -> {len(code_blocks)} unique inner blocks")

        # 3. 各ブロックのテキスト抽出処理（前回と同じ）
        for block in code_blocks:
            lines = []
            rows = block.find_all('tr')
            
            if rows:
                for row in rows:
                    # 行単位の重複排除: 中にtableを持つ行（外枠の行）はスキップ
                    if row.find('table'):
                        continue
                    lines.append(row.get_text(separator="", strip=False))
            else:
                for br in block.find_all("br"): br.replace_with("\n")
                text = block.get_text(separator="", strip=False)
                lines = text.splitlines()
            
            raw_block_text = "\n".join(lines)

            # 純粋な定義のみ抽出
            clean_code = self.extract_pure_definitions(raw_block_text)

            if len(clean_code) < 5: continue

            # コンテキスト取得
            context_text = ""
            prev = block.find_previous_sibling(['p', 'div', 'h2', 'h3', 'h4'])
            if not prev and block.parent:
                prev = block.parent.find_previous_sibling(['p', 'div', 'h2', 'h3', 'h4'])
            if prev:
                context_text = re.sub(r'\s+', ' ', prev.get_text(separator=" ")).strip()

            extracted_data.append({
                "instruction": context_text,
                "output": clean_code,
                "source": source_url
            })

        return extracted_data
    
    def get_internal_links(self, soup, current_url):
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(current_url, href)
            clean_url, _ = urldefrag(full_url)
            parsed = urlparse(clean_url)
            current_clean, _ = urldefrag(current_url)

            # フィルタリング: ドメイン一致 + パス制限(/guide/) + HTML + その他除外
            if (parsed.netloc == self.base_domain and 
                parsed.path.startswith(self.target_prefix) and # <--- ここで制限
                parsed.path.endswith('.html') and
                'search' not in parsed.path and
                clean_url != current_clean):
                
                links.add(clean_url)
                
        return list(links)

    def run(self):
        print(f"Start crawling from: {self.start_url} (Restricted to {self.target_prefix})")
        open(self.output_file, 'w').close()

        count = 0
        url = self.queue.popleft()
        soup = self.fetch_page(url)
        new_links = self.get_internal_links(soup, url)
        for link in new_links:
            self.queue.append(link)
        while self.queue and count < self.max_pages:
            url = self.queue.popleft()
            if url in self.visited: continue
            
            print(f"[{count+1}/{self.max_pages}] Processing: {url}")
            soup = self.fetch_page(url)
            self.visited.add(url)

            if soup:
                pairs = self.extract_pairs(soup, url)
                if pairs:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        for entry in pairs:
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    print(f"  -> Saved {len(pairs)} pairs.")
            
            count += 1
            time.sleep(1.0)
        
        print("Done.")

if __name__ == "__main__":
    start_url = "https://docs.racket-lang.org/guide/index.html"
    scraper = RacketGuideScraper(start_url, max_pages=1000) # Guide全体を取るためページ数を増やす
    scraper.run()