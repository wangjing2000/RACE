from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

class NERExtractor:
    def __init__(self, model_dir: str | Path, language: str = "zh"):
        self.language = language
        model_dir = str(model_dir)

        if language == "zh":
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
        else:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)

        self.model.eval()

    def extract(self, text: str) -> List[Tuple[str, str]]:
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = []
        current_entity = ""
        current_label = "O"

        for token, pred in zip(tokens, predictions):
            label = self.model.config.id2label.get(int(pred), "O")
            if label.startswith("B-") or label.startswith("I-"):
                entity_type = label[2:]
                if label.startswith("B-") or entity_type != current_label:
                    if current_entity:
                        entities.append((current_entity, current_label))
                    current_entity = token.replace("##", "")
                    current_label = entity_type
                else:
                    current_entity += token.replace("##", "")
            else:
                if current_entity:
                    entities.append((current_entity, current_label))
                    current_entity = ""
                    current_label = "O"

        if current_entity:
            entities.append((current_entity, current_label))

        return entities

class KeywordExtractor:
    def __init__(self, language: str = "zh"):
        self.language = language

    def extract(self, text: str, top_k: int = 10) -> List[str]:
        if self.language == "zh":
            import jieba
            import jieba.analyse
            keywords = jieba.analyse.extract_tags(text, topK=top_k)
        else:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            from collections import Counter
            counter = Counter(words)
            keywords = [w for w, _ in counter.most_common(top_k)]

        return keywords

class SemanticUnitExtractor:
    def __init__(self, language: str = "zh"):
        self.language = language
        self.ner = None
        self.keyword = KeywordExtractor(language)

    def extract(self, text: str, top_k: int = 10) -> List[str]:
        keywords = self.keyword.extract(text, top_k)
        return keywords

    def cache_units(
        self,
        json_path: str | Path,
        cache_dir: str | Path,
        top_k: int = 10,
    ) -> None:
        json_path = Path(json_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for sample in tqdm(samples, desc="Caching semantic units"):
            sample_id = sample.get("id", "")
            cache_path = cache_dir / f"{sample_id}_units.json"
            if cache_path.exists():
                continue

            text = sample.get("text", "")
            units = self.extract(text, top_k)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(units, f, ensure_ascii=False, indent=2)