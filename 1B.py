import fitz
import json
import re
from collections import Counter, defaultdict
import argparse
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import statistics

# =====================================================================================
#  COMPONENT 1: PDF Outline Extractor (Final Architecture)
# =====================================================================================
class PDFOutlineExtractor:
    """
    Extracts a hierarchical outline using an advanced, multi-pass structural
    analysis pipeline for high-precision heading detection.
    """

    def __init__(self, pdf_path: str):
        try:
            if len(pdf_path) > 260 and re.match(r'^[a-zA-Z]:\\', pdf_path):
                 pdf_path = "\\\\?\\" + pdf_path
            self.doc = fitz.open(pdf_path)
        except Exception as e:
            raise FileNotFoundError(f"Error opening or reading PDF file: {e}")

    def _is_bold_by_name(self, font_name: str) -> bool:
        return any(x in font_name.lower() for x in ['bold', 'black', 'heavy', 'condb', 'cbi'])

    def _get_text_blocks(self):
        """Pass 1: Reconstruct the document into logical text blocks."""
        blocks = []
        for page in self.doc:
            for block in page.get_text("dict")["blocks"]:
                if block['type'] == 0: # Text block
                    block_text = ""
                    span_styles = []
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                            span_styles.append((round(span['size']), self._is_bold_by_name(span['font'])))
                    
                    if not block_text.strip() or not re.search('[a-zA-Z]', block_text): continue
                    
                    if not span_styles: continue
                    dominant_style = Counter(span_styles).most_common(1)[0][0]
                    
                    blocks.append({
                        'text': block_text.strip(),
                        'style': dominant_style,
                        'bbox': block['bbox'],
                        'page_num': page.number + 1,
                        'num_lines': len(block['lines']),
                        'num_words': len(block_text.split())
                    })
        return blocks

    def _find_body_style(self, blocks):
        """Pass 2: Identify the primary body text style based on total word count."""
        style_word_counts = defaultdict(int)
        for block in blocks:
            if block['num_lines'] > 2 or block['num_words'] > 20:
                 style_word_counts[block['style']] += block['num_words']
        
        if not style_word_counts:
            style_freq = Counter(b['style'] for b in blocks)
            if not style_freq: return None
            return style_freq.most_common(1)[0][0]

        return max(style_word_counts, key=style_word_counts.get)

    def get_outline(self) -> dict:
        """Orchestrates the new multi-pass pipeline to extract the outline."""
        title = self._extract_title()
        
        toc = self.doc.get_toc()
        if toc:
            outline = [{"level": f"H{level}", "text": text.strip(), "page_num": page, "bbox": None} for level, text, page in toc if 1 <= level <= 4]
            outline = [h for h in outline if re.search('[a-zA-Z]', h['text'])]
            if outline:
                 return {"title": title, "outline": outline}

        all_blocks = self._get_text_blocks()
        if not all_blocks:
            return {"title": title, "outline": []}
            
        body_style = self._find_body_style(all_blocks)
        if not body_style:
            return {"title": title, "outline": []}

        heading_blocks = []
        for block in all_blocks:
            if block['num_words'] > 30 or block['num_lines'] > 3:
                continue
            
            is_candidate = block['style'][0] > body_style[0] or \
                           (block['style'][0] == body_style[0] and block['style'][1] and not body_style[1])
            if not is_candidate:
                continue

            text = block['text'].strip()
            if re.search(r'\.{4,}', text) or text.endswith(('.', ',', ';', ':')):
                continue
            if re.match(r'^\s*([•*-]|[a-zA-Z\d]+\))\s+', text):
                continue

            heading_blocks.append(block)

        if not heading_blocks:
            return {"title": title, "outline": []}

        heading_styles = set(b['style'] for b in heading_blocks)
        size_groups = defaultdict(list)
        for size, bold in heading_styles:
            size_groups[size].append((size, bold))
        
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        style_to_level = {}
        level_map = ['H1', 'H2', 'H3', 'H4']
        for i, size in enumerate(sorted_sizes):
            if i < len(level_map):
                level = level_map[i]
                for style in sorted(size_groups[size], key=lambda s: s[1], reverse=True):
                    style_to_level[style] = level

        final_outline = []
        list_item_pattern = re.compile(r'^\s*(\d+(\.\d+)*)\s+')
        for block in heading_blocks:
            if block['style'] in style_to_level:
                level = style_to_level[block['style']]
                text = ' '.join(block['text'].split())
                
                match = list_item_pattern.match(text)
                if match:
                    dot_count = match.group(1).count('.')
                    level = f"H{dot_count + 1}"

                if level == 'H1' and block['page_num'] == 1 and text == title:
                    continue

                final_outline.append({'text': text, 'level': level, 'page_num': block['page_num'], 'bbox': block['bbox']})
        
        return {"title": title, "outline": sorted(final_outline, key=lambda x: (x['page_num'], x['bbox'][1]))}

    def _extract_title(self) -> str:
        if self.doc.metadata and (title := self.doc.metadata.get("title", "").strip()):
            if len(title) > 4 and not re.search(r'\.(pdf|docx?|pptx?|xlsx?|cdr)$', title, re.I) and "Microsoft Word" not in title:
                return title
        if not self.doc or self.doc.page_count == 0: return ""
        first_page = self.doc[0]
        top_rect = fitz.Rect(0, 0, first_page.rect.width, first_page.rect.height * 0.4)
        blocks = first_page.get_text("dict", clip=top_rect).get('blocks', [])
        font_sizes = defaultdict(list)
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    line_text = " ".join(s['text'].strip() for s in line['spans'] if s['text'].strip()).strip()
                    if line_text and re.search('[a-zA-Z]', line_text) and len(line_text.split()) < 20:
                        if line['spans']:
                             avg_size = round(sum(s['size'] for s in line['spans']) / len(line['spans']))
                             font_sizes[avg_size].append(line_text)
        if font_sizes:
            max_size = max(font_sizes.keys())
            return " ".join(font_sizes[max_size])
        return ""

# =====================================================================================
#  COMPONENT 2: Document Sectionizer
# =====================================================================================
class DocumentSectionizer:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.outline = PDFOutlineExtractor(pdf_path).get_outline()['outline']

    def get_sections(self) -> list:
        sections = []
        for i, heading in enumerate(self.outline):
            if 'bbox' not in heading or not heading['bbox']: continue
            start_page = heading['page_num'] - 1
            start_y = heading['bbox'][3] 
            if i + 1 < len(self.outline) and 'bbox' in self.outline[i+1] and self.outline[i+1]['bbox']:
                next_heading = self.outline[i+1]
                end_page = next_heading['page_num'] - 1
                end_y = next_heading['bbox'][1]
            else:
                end_page = len(self.doc) - 1
                end_y = self.doc[end_page].rect.height
            content = ""
            for page_num in range(start_page, end_page + 1):
                page = self.doc[page_num]
                clip_y_start = start_y if page_num == start_page else 0
                clip_y_end = end_y if page_num == end_page else page.rect.height
                if clip_y_start < clip_y_end:
                    clip_rect = fitz.Rect(0, clip_y_start, page.rect.width, clip_y_end)
                    content += page.get_text(clip=clip_rect)
            
            # Clean the extracted text to make it more readable
            cleaned_content = re.sub(r'(?<!\n)\n(?!\n)', ' ', content)
            cleaned_content = re.sub(r' \n', '\n', cleaned_content)
            cleaned_content = re.sub(r'\n{2,}', '\n', cleaned_content)
            cleaned_content = cleaned_content.strip()

            sections.append({'section_title': heading['text'], 'page_number': heading['page_num'], 'content': f"{heading['text']}\n{cleaned_content}"})
        return sections

# =====================================================================================
#  COMPONENT 3: Semantic Ranker
# =====================================================================================
class SemanticRanker:
    def __init__(self, model_path='all-MiniLM-L6-v2-local'):
        try:
            self.model = SentenceTransformer(model_path)
        except Exception as e:
            raise IOError(f"Failed to load model. Ensure it's downloaded. Error: {e}")
    def rank_sections(self, persona: str, job_to_be_done: str, all_sections: list) -> list:
        if not all_sections: return []
        query = f"User Persona: {persona}. Task: {job_to_be_done}"
        query_embedding = self.model.encode([query])
        section_contents = [section['content'] for section in all_sections]
        section_embeddings = self.model.encode(section_contents)
        similarities = cosine_similarity(query_embedding, section_embeddings)[0]
        for i, section in enumerate(all_sections):
            section['relevance_score'] = similarities[i]
        return sorted(all_sections, key=lambda x: x['relevance_score'], reverse=True)

# =====================================================================================
#  MAIN EXECUTION BLOCK
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence System.")
    parser.add_argument("collection_dir", type=str, help="Path to the collection directory (e.g., 'Input/').")
    args = parser.parse_args()
    print(f"Starting analysis for collection: {args.collection_dir}")
    input_json_path = os.path.join(args.collection_dir, 'challenge1b_input.json')
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f: config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading input JSON: {e}")
        return
    persona = config.get('persona', {}).get('role', '')
    job_to_be_done = config.get('job_to_be_done', {}).get('task', '')
    documents = config.get('documents', [])
    pdf_dir = os.path.join(args.collection_dir, 'PDFs')
    pdf_paths = [os.path.join(pdf_dir, doc['filename']) for doc in documents if 'filename' in doc]
    all_sections = []
    for pdf_path in pdf_paths:
        doc_name = os.path.basename(pdf_path)
        print(f"  - Processing: {doc_name}")
        if not os.path.exists(pdf_path):
            print(f"    - Warning: File not found, skipping: {pdf_path}")
            continue
        try:
            sectionizer = DocumentSectionizer(pdf_path)
            sections = sectionizer.get_sections()
            for section in sections:
                section['document'] = doc_name
            all_sections.extend(sections)
        except Exception as e:
            print(f"    - Could not process {doc_name}. Error: {e}")
    ranker = SemanticRanker()
    ranked_sections = ranker.rank_sections(persona, job_to_be_done, all_sections)
    output_data = {
        "metadata": {"input_documents": [doc['filename'] for doc in documents], "persona": persona, "job_to_be_done": job_to_be_done},
        "extracted_sections": [], "subsection_analysis": []
    }
    for i, section in enumerate(ranked_sections[:10]):
        output_data["extracted_sections"].append({"document": section['document'], "section_title": section['section_title'], "importance_rank": i + 1, "page_number": section['page_number']})
    for section in ranked_sections[:5]:
        output_data["subsection_analysis"].append({"document": section['document'], "refined_text": section['content'], "page_number": section['page_number']})
    output_dir = os.path.join(args.collection_dir, '../Output')
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, 'challenge1b_output.json')

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Analysis complete. Output saved to {output_json_path}")

if __name__ == "__main__":
    main()
