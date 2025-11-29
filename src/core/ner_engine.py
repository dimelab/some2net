"""Named Entity Recognition engine with caching and language detection."""
from typing import List, Dict, Optional
import torch
from transformers import pipeline, AutoModelForTokenClassification, XLMRobertaTokenizer
from tqdm import tqdm
import hashlib
import json
from pathlib import Path
from diskcache import Cache
from langdetect import detect, LangDetectException


class NEREngine:
    """Extract named entities from text with caching and language detection."""
    
    def __init__(
        self,
        model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85,
        cache_dir: str = "./cache/ner_results",
        enable_cache: bool = True
    ):
        """
        Initialize NER engine with caching.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            confidence_threshold: Minimum confidence score for entities
            cache_dir: Directory for caching NER results
            enable_cache: Enable/disable result caching
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.enable_cache = enable_cache
        
        # Initialize cache
        if enable_cache:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self.cache = Cache(cache_dir)
        else:
            self.cache = None
        
        # Determine device with better CUDA error handling
        if device is None:
            try:
                # Check if CUDA is available and properly initialized
                if torch.cuda.is_available():
                    # Try to initialize CUDA
                    torch.cuda.init()
                    torch.cuda.current_device()
                    self.device = 0
                    print("✅ GPU (CUDA) detected and initialized")
                else:
                    self.device = -1
                    print("ℹ️  No GPU detected, using CPU")
            except Exception as e:
                print(f"⚠️  CUDA initialization failed: {e}")
                print("ℹ️  Falling back to CPU mode")
                self.device = -1
        elif device == "cuda":
            try:
                if not torch.cuda.is_available():
                    print("⚠️  Warning: CUDA requested but not available, falling back to CPU")
                    self.device = -1
                else:
                    torch.cuda.init()
                    torch.cuda.current_device()
                    self.device = 0
                    print("✅ GPU (CUDA) initialized")
            except Exception as e:
                print(f"⚠️  CUDA initialization failed: {e}")
                print("ℹ️  Falling back to CPU mode")
                self.device = -1
        else:
            self.device = -1
            print("ℹ️  Using CPU mode")
        
        # Load model
        print(f"🔄 Loading NER model: {model_name}")
        print(f"📱 Device: {'GPU (CUDA)' if self.device >= 0 else 'CPU'}")

        try:
            # Explicitly load tokenizer using the slow XLMRobertaTokenizer class
            # This avoids the fast tokenizer bug with vocab_file
            print("📥 Loading tokenizer (slow tokenizer)...")
            tokenizer = XLMRobertaTokenizer.from_pretrained(
                model_name,
                add_prefix_space=True
            )

            print("📥 Loading model...")
            model = AutoModelForTokenClassification.from_pretrained(model_name)

            # Create pipeline with explicit tokenizer and model
            print("🔧 Creating NER pipeline...")
            self.nlp = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=self.device
            )
            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Troubleshooting steps:")
            print(f"1. Clear cache: rm -rf ~/.cache/huggingface/")
            print(f"2. Upgrade transformers: pip install --upgrade transformers")
            print(f"3. Try manual download:")
            print(f"   python -c \"from transformers import AutoTokenizer, AutoModelForTokenClassification;")
            print(f"   AutoTokenizer.from_pretrained('{model_name}', use_fast=False);")
            print(f"   AutoModelForTokenClassification.from_pretrained('{model_name}')\"")
            print(f"\nSee TROUBLESHOOTING_MODEL_LOAD.md for more details.")
            raise RuntimeError(f"Failed to load NER model: {e}")

        if enable_cache:
            print(f"💾 Cache enabled at: {cache_dir}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            ISO language code (e.g., 'en', 'da', 'unknown')
        """
        # Handle None and empty strings
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return 'unknown'

        try:
            # Try to detect language
            lang = detect(text)
            return lang
        except LangDetectException:
            # If detection fails, return unknown
            return 'unknown'
        except Exception:
            # Catch any other errors (like AttributeError on None)
            return 'unknown'
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key (hash)
        """
        # Create key from text + model + threshold
        key_str = f"{text}|{self.model_name}|{self.confidence_threshold}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def extract_entities_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        detect_languages: bool = True
    ) -> tuple[List[List[Dict]], List[str]]:
        """
        Extract entities from multiple texts with caching.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            detect_languages: Detect language for each text
            
        Returns:
            Tuple of (entity lists, detected languages)
        """
        all_results = []
        all_languages = []
        cache_hits = 0
        cache_misses = 0
        
        # Check cache first
        texts_to_process = []
        texts_indices = []
        
        for i, text in enumerate(texts):
            # Handle None or non-string values
            if text is None or not isinstance(text, str):
                text = ""
                all_results.append([])
                all_languages.append('unknown')
                continue

            # Handle empty strings
            if len(text.strip()) == 0:
                all_results.append([])
                all_languages.append('unknown')
                continue

            # Detect language if requested
            if detect_languages:
                lang = self.detect_language(text)
                all_languages.append(lang)
            else:
                all_languages.append('unknown')

            # Check cache
            if self.enable_cache and self.cache is not None:
                cache_key = self._get_cache_key(text)
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    all_results.append(cached_result)
                    cache_hits += 1
                else:
                    all_results.append(None)  # Placeholder
                    texts_to_process.append(text)
                    texts_indices.append(i)
                    cache_misses += 1
            else:
                all_results.append(None)
                texts_to_process.append(text)
                texts_indices.append(i)
        
        if self.enable_cache:
            print(f"💾 Cache: {cache_hits} hits, {cache_misses} misses")
        
        # Process uncached texts
        if texts_to_process:
            iterator = range(0, len(texts_to_process), batch_size)
            if show_progress:
                iterator = tqdm(
                    iterator, 
                    desc="Extracting entities",
                    total=(len(texts_to_process) + batch_size - 1) // batch_size
                )
            
            for i in iterator:
                batch = texts_to_process[i:i + batch_size]
                batch_indices = texts_indices[i:i + batch_size]
                
                try:
                    # Run NER on batch
                    batch_results = self.nlp(batch)
                    
                    # Filter by confidence and clean results
                    for j, (result, original_idx) in enumerate(zip(batch_results, batch_indices)):
                        cleaned = self._clean_entities(result)
                        all_results[original_idx] = cleaned
                        
                        # Cache result
                        if self.enable_cache and self.cache is not None:
                            cache_key = self._get_cache_key(texts_to_process[i + j])
                            self.cache.set(cache_key, cleaned)
                    
                    # Clear GPU cache
                    if self.device >= 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"❌ Error processing batch: {e}")
                    # Add empty results for failed batch
                    for original_idx in batch_indices:
                        if all_results[original_idx] is None:
                            all_results[original_idx] = []
        
        return all_results, all_languages
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from single text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries
        """
        results, _ = self.extract_entities_batch([text], show_progress=False)
        return results[0]
    
    def _clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Clean and filter entity results.

        Args:
            entities: Raw entity results from pipeline

        Returns:
            Cleaned entity list
        """
        # First pass: filter and standardize
        filtered = []

        for entity in entities:
            # Filter by confidence
            if entity.get('score', 0) < self.confidence_threshold:
                continue

            # Standardize entity type labels
            entity_type = entity.get('entity_group', '')

            # Map to standard labels
            if entity_type in ['PER', 'PERSON']:
                entity_type = 'PER'
            elif entity_type in ['LOC', 'LOCATION']:
                entity_type = 'LOC'
            elif entity_type in ['ORG', 'ORGANIZATION']:
                entity_type = 'ORG'
            else:
                # Skip MISC and other types
                continue

            entity_text = entity.get('word', '').strip()
            if not entity_text:
                continue

            # Get start/end positions, use defaults if None
            start_pos = entity.get('start', 0)
            end_pos = entity.get('end', 0)

            filtered.append({
                'text': entity_text,
                'type': entity_type,
                'score': entity['score'],
                'start': start_pos,
                'end': end_pos
            })

        # Second pass: merge adjacent entities of the same type
        # This fixes tokenization issues like "Ch" + "ristiansborg" -> "Christiansborg"
        # Strategy: Merge BEFORE filtering, so short prefixes get merged
        if not filtered:
            return []

        # Sort by start position to ensure proper order
        filtered.sort(key=lambda x: (x['start'], x['end']))

        merged = []
        i = 0
        while i < len(filtered):
            current = filtered[i]

            # Try merging with next entities if they're adjacent and same type
            while i + 1 < len(filtered):
                next_entity = filtered[i + 1]

                try:
                    # Calculate distance between entities
                    distance = next_entity['start'] - current['end']

                    # Merge if same type and adjacent (within 2 chars gap)
                    if (next_entity['type'] == current['type'] and distance <= 2):
                        # Merge the entities
                        gap = ' ' if distance > 0 else ''
                        current = {
                            'text': current['text'] + gap + next_entity['text'],
                            'type': current['type'],
                            'score': (current['score'] + next_entity['score']) / 2,
                            'start': current['start'],
                            'end': next_entity['end']
                        }
                        i += 1  # Skip the next entity since we merged it
                    else:
                        break
                except (TypeError, KeyError):
                    # Skip merging if position data is invalid
                    break

            # Only keep entities with 2+ characters (after merging)
            if len(current['text'].strip()) >= 2:
                merged.append(current)

            i += 1

        return merged
    
    def clear_cache(self) -> None:
        """Clear all cached NER results."""
        if self.cache is not None:
            self.cache.clear()
            print("🗑️  Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self.cache is not None:
            return {
                'size': len(self.cache),
                'size_bytes': self.cache.volume()
            }
        return {'size': 0, 'size_bytes': 0}


# Example usage
if __name__ == "__main__":
    # Initialize engine with caching
    engine = NEREngine(enable_cache=True)
    
    # Single text
    text = "John Smith works at Microsoft in Copenhagen."
    entities = engine.extract_entities(text)
    print("Entities:", entities)
    
    # Batch processing with language detection
    texts = [
        "Angela Merkel visited Paris last week.",
        "Apple Inc. announced a new product in California.",
        "Statsministeren mødtes med embedsmænd i København."  # Danish
    ]
    results, languages = engine.extract_entities_batch(texts, detect_languages=True)
    
    for text, entities, lang in zip(texts, results, languages):
        print(f"\nText: {text}")
        print(f"Language: {lang}")
        print(f"Entities: {entities}")
    
    # Cache stats
    print(f"\nCache stats: {engine.get_cache_stats()}")
    
    # Test cache hit
    print("\n--- Testing cache hit ---")
    results2, _ = engine.extract_entities_batch(texts)  # Should hit cache
