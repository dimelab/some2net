"""
Entity Linking Module using mGENRE (Multilingual Autoregressive Entity Linking)

Links extracted named entities to Wikipedia/Wikidata entries for improved
cross-language entity resolution and disambiguation.

Model: facebook/mgenre-wiki
Paper: "Multilingual Autoregressive Entity Linking" (De Cao et al., 2021)
"""

from typing import List, Dict, Optional, Tuple, Set
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import hashlib
import re
from pathlib import Path
from diskcache import Cache
import logging
import requests
from urllib.parse import quote
import numpy as np
from collections import defaultdict, Counter
import json

# Optional Phase 3 dependency
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EntityLinker:
    """
    Link named entities to Wikipedia/Wikidata using mGENRE model.

    Features:
    - Multilingual entity linking (105 languages)
    - Wikipedia page disambiguation
    - Wikidata ID extraction
    - Cross-language entity resolution
    - Disk caching for efficiency
    - Advanced disambiguation with context embeddings (Phase 3)
    - Entity relationship extraction (Phase 3)
    - Custom knowledge base support (Phase 3)
    - Entity description retrieval from Wikidata (Phase 4)
    - Typed relationship extraction (works-for, located-in, part-of) (Phase 4)
    - Document-level context for disambiguation (Phase 4)
    """

    def __init__(
        self,
        model_name: str = "facebook/mgenre-wiki",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
        top_k: int = 5,
        cache_dir: str = "./cache/entity_links",
        enable_cache: bool = True,
        enable_advanced_disambiguation: bool = False,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        custom_kb_path: Optional[str] = None,
        enable_entity_descriptions: bool = False,  # Phase 4
        enable_typed_relationships: bool = False,  # Phase 4
        use_document_context: bool = False  # Phase 4
    ):
        """
        Initialize entity linker with mGENRE model.

        Args:
            model_name: HuggingFace model name for entity linking
            device: Device to use ('cuda', 'cpu', or None for auto)
            confidence_threshold: Minimum confidence for accepting links
            top_k: Number of candidate links to retrieve
            cache_dir: Directory for caching linking results
            enable_cache: Enable/disable result caching
            enable_advanced_disambiguation: Enable context-based disambiguation (Phase 3)
            embedding_model: Sentence embedding model for disambiguation
            custom_kb_path: Path to custom entity knowledge base JSON file
            enable_entity_descriptions: Fetch entity descriptions from Wikidata (Phase 4)
            enable_typed_relationships: Extract typed relationships (Phase 4)
            use_document_context: Use document-level context for disambiguation (Phase 4)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.enable_cache = enable_cache
        self.enable_advanced_disambiguation = enable_advanced_disambiguation

        # Phase 3: Custom knowledge base
        self.custom_kb = self._load_custom_kb(custom_kb_path) if custom_kb_path else {}

        # Phase 3: Entity co-occurrence tracking
        self.entity_cooccurrence: Dict[str, Counter] = defaultdict(Counter)

        # Phase 4: Entity descriptions and advanced features
        self.enable_entity_descriptions = enable_entity_descriptions
        self.enable_typed_relationships = enable_typed_relationships
        self.use_document_context = use_document_context
        self.entity_descriptions_cache: Dict[str, str] = {}  # Cache for Wikidata descriptions
        self.document_context: Optional[str] = None  # Current document context

        # Initialize cache
        if enable_cache:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self.cache = Cache(cache_dir)
        else:
            self.cache = None

        # Determine device
        if device is None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.init()
                    torch.cuda.current_device()
                    self.device = "cuda"
                    print("✅ GPU (CUDA) detected for entity linking")
                else:
                    self.device = "cpu"
                    print("ℹ️  No GPU detected for entity linking, using CPU")
            except Exception as e:
                print(f"⚠️  CUDA initialization failed: {e}")
                print("ℹ️  Falling back to CPU for entity linking")
                self.device = "cpu"
        elif device == "cuda":
            try:
                if not torch.cuda.is_available():
                    print("⚠️  Warning: CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
                else:
                    torch.cuda.init()
                    torch.cuda.current_device()
                    self.device = "cuda"
                    print("✅ GPU (CUDA) initialized for entity linking")
            except Exception as e:
                print(f"⚠️  CUDA initialization failed: {e}")
                print("ℹ️  Falling back to CPU for entity linking")
                self.device = "cpu"
        else:
            self.device = "cpu"
            print("ℹ️  Using CPU for entity linking")

        # Load model
        print(f"🔄 Loading Entity Linking model: {model_name}")
        print(f"📱 Device: {self.device.upper()}")

        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            print("📥 Loading model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            print("✅ Entity linking model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading entity linking model: {e}")
            print("\n💡 Troubleshooting steps:")
            print(f"1. Install required dependencies: pip install sentencepiece")
            print(f"2. Clear cache: rm -rf ~/.cache/huggingface/")
            print(f"3. Check model availability at: https://huggingface.co/{model_name}")
            raise RuntimeError(f"Failed to load entity linking model: {e}")

        if enable_cache:
            print(f"💾 Entity linking cache enabled at: {cache_dir}")

        # Phase 3: Initialize sentence embedding model for advanced disambiguation
        self.embedder = None
        if enable_advanced_disambiguation:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence-transformers not installed. Advanced disambiguation disabled.")
                print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")
                print("ℹ️  Advanced disambiguation disabled")
                self.enable_advanced_disambiguation = False
            else:
                try:
                    print(f"🔄 Loading embedding model for advanced disambiguation: {embedding_model}")
                    self.embedder = SentenceTransformer(embedding_model, device=self.device)
                    print("✅ Embedding model loaded successfully!")
                except Exception as e:
                    logger.warning(f"Failed to load embedding model: {e}")
                    print(f"⚠️  Advanced disambiguation disabled due to model loading error")
                    self.enable_advanced_disambiguation = False

        if self.custom_kb:
            print(f"📚 Loaded custom knowledge base with {len(self.custom_kb)} entities")

    def _load_custom_kb(self, kb_path: str) -> Dict[str, Dict]:
        """
        Load custom entity knowledge base from JSON file.

        Format:
        {
            "entity_text": {
                "canonical_name": "Official Name",
                "wikidata_id": "Q12345",
                "aliases": ["alias1", "alias2"],
                "type": "PER/LOC/ORG"
            }
        }

        Args:
            kb_path: Path to JSON file

        Returns:
            Dictionary mapping entity text to entity info
        """
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
                # Build reverse index for aliases
                expanded_kb = {}
                for entity_text, info in kb.items():
                    expanded_kb[entity_text.lower()] = info
                    # Add aliases as additional lookup keys
                    for alias in info.get('aliases', []):
                        expanded_kb[alias.lower()] = info
                return expanded_kb
        except Exception as e:
            logger.warning(f"Failed to load custom knowledge base from {kb_path}: {e}")
            return {}

    def _get_cache_key(self, entity_text: str, entity_type: str, language: str = "en") -> str:
        """
        Generate cache key for entity linking.

        Args:
            entity_text: Entity text
            entity_type: Entity type (PER/LOC/ORG)
            language: Source language code

        Returns:
            Cache key (hash)
        """
        key_str = f"{entity_text}|{entity_type}|{language}|{self.model_name}|{self.top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def link_entity(
        self,
        entity_text: str,
        entity_type: str = "ENTITY",
        language: str = "en",
        context: Optional[str] = None,
        co_entities: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Link a single entity to Wikipedia/Wikidata.

        Args:
            entity_text: Entity text to link
            entity_type: Entity type (PER/LOC/ORG)
            language: Source language code (e.g., 'en', 'da')
            context: Optional context sentence for disambiguation
            co_entities: List of other entities mentioned in same context (Phase 3)

        Returns:
            Dictionary with linking information or None if linking fails
            {
                'wikipedia_title': str,
                'wikidata_id': str,
                'wikipedia_url': str,
                'canonical_name': str,
                'language_variants': dict,
                'linking_confidence': float,
                'candidates': list,
                'disambiguation_method': str  # Phase 3
            }
        """
        # Phase 3: Check custom knowledge base first
        if self.custom_kb:
            kb_result = self._lookup_custom_kb(entity_text, entity_type)
            if kb_result:
                return kb_result

        # Check cache
        if self.enable_cache and self.cache is not None:
            cache_key = self._get_cache_key(entity_text, entity_type, language)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Prepare input text
        # mGENRE expects format: "[START] entity_text [END] optional_context"
        if context:
            input_text = f"[START] {entity_text} [END] {context}"
        else:
            input_text = f"[START] {entity_text} [END]"

        try:
            # Tokenize
            inputs = self.tokenizer(
                [input_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate candidates
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_beams=self.top_k,
                    num_return_sequences=self.top_k,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_length=128
                )

            # Decode candidates
            candidates = self.tokenizer.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )

            # Calculate confidence scores from generation scores
            # Use the average log probability as a proxy for confidence
            if hasattr(outputs, 'sequences_scores'):
                scores = torch.softmax(outputs.sequences_scores, dim=0).cpu().numpy()
            else:
                # Fallback: uniform distribution if scores not available
                scores = [1.0 / len(candidates)] * len(candidates)

            # Process candidates
            candidate_list = []
            for i, (candidate, score) in enumerate(zip(candidates, scores)):
                # Parse candidate: format is "Entity Name >> language_code"
                parsed = self._parse_candidate(candidate)
                if parsed:
                    candidate_list.append({
                        'rank': i + 1,
                        'wikipedia_title': parsed['title'],
                        'language': parsed['language'],
                        'confidence': float(score)
                    })

            # Phase 3/4: Advanced disambiguation if enabled
            effective_context = context
            # Phase 4: Use document context if available and enabled
            if self.use_document_context and self.document_context:
                effective_context = self.document_context[:1000]  # Use first 1000 chars

            if self.enable_advanced_disambiguation and effective_context and candidate_list:
                best = self._advanced_disambiguation(
                    entity_text, candidate_list, effective_context, co_entities
                )
                disambiguation_method = "advanced" if not self.use_document_context else "document_level"
            elif candidate_list:
                best = candidate_list[0]
                disambiguation_method = "baseline"
            else:
                best = None
                disambiguation_method = "none"

            # Get best candidate
            if best and best['confidence'] >= self.confidence_threshold:
                # Extract Wikidata ID (requires additional lookup - simplified for now)
                wikidata_id = self._get_wikidata_id(best['wikipedia_title'], best['language'])

                # Build result
                result = {
                    'wikipedia_title': best['wikipedia_title'],
                    'wikidata_id': wikidata_id,
                    'wikipedia_url': self._build_wikipedia_url(best['wikipedia_title'], best['language']),
                    'canonical_name': best['wikipedia_title'].replace('_', ' '),
                    'language_variants': self._extract_language_variants(candidate_list),
                    'linking_confidence': best['confidence'],
                    'candidates': candidate_list[:3],  # Top 3 candidates
                    'disambiguation_method': disambiguation_method  # Phase 3
                }

                # Phase 4: Add entity description if enabled
                if self.enable_entity_descriptions and wikidata_id:
                    description = self._get_entity_description(wikidata_id, language)
                    if description:
                        result['description'] = description

                # Phase 3: Track co-occurrence for future disambiguation
                if co_entities and wikidata_id:
                    for co_entity in co_entities:
                        if co_entity != entity_text:
                            self.entity_cooccurrence[wikidata_id][co_entity] += 1

                # Cache result
                if self.enable_cache and self.cache is not None:
                    cache_key = self._get_cache_key(entity_text, entity_type, language)
                    self.cache.set(cache_key, result)

                return result
            else:
                # Confidence too low or no candidates
                if self.enable_cache and self.cache is not None:
                    cache_key = self._get_cache_key(entity_text, entity_type, language)
                    self.cache.set(cache_key, None)
                return None

        except Exception as e:
            logger.warning(f"Error linking entity '{entity_text}': {e}")
            return None

    def link_entities_batch(
        self,
        entities: List[Dict],
        batch_size: int = 16,
        default_language: str = "en",
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Link multiple entities with batch processing.

        Args:
            entities: List of entity dictionaries with 'text', 'type', 'score'
                     Can optionally include 'context' and 'language'
            batch_size: Number of entities to process at once
            default_language: Default language if not specified per entity
            show_progress: Show progress bar

        Returns:
            List of entity dictionaries enhanced with linking information
        """
        enhanced_entities = []
        cache_hits = 0
        cache_misses = 0
        linked_count = 0

        for entity in entities:
            entity_text = entity.get('text', '').strip()
            entity_type = entity.get('type', 'ENTITY')
            context = entity.get('context', None)
            language = entity.get('language', default_language)

            if not entity_text:
                enhanced_entities.append(entity)
                continue

            # Attempt to link entity
            link_result = self.link_entity(
                entity_text,
                entity_type,
                language,
                context
            )

            # Enhance entity with linking information
            enhanced_entity = entity.copy()

            if link_result:
                enhanced_entity.update({
                    'wikidata_id': link_result['wikidata_id'],
                    'wikipedia_url': link_result['wikipedia_url'],
                    'wikipedia_title': link_result['wikipedia_title'],
                    'canonical_name': link_result['canonical_name'],
                    'language_variants': link_result['language_variants'],
                    'linking_confidence': link_result['linking_confidence'],
                    'is_linked': True
                })
                linked_count += 1
            else:
                enhanced_entity['is_linked'] = False

            enhanced_entities.append(enhanced_entity)

        if self.enable_cache:
            print(f"🔗 Entity linking: {linked_count}/{len(entities)} linked successfully")

        return enhanced_entities

    def _parse_candidate(self, candidate: str) -> Optional[Dict]:
        """
        Parse mGENRE candidate output.

        Format: "Entity_Name >> language_code"
        Example: "Copenhagen >> en"

        Args:
            candidate: Raw candidate string

        Returns:
            Parsed dictionary or None
        """
        # Pattern: "Title >> lang"
        match = re.match(r'^(.+?)\s*>>\s*([a-z]{2})$', candidate.strip())
        if match:
            return {
                'title': match.group(1).strip(),
                'language': match.group(2).strip()
            }

        # Fallback: assume English if no language specified
        if candidate.strip():
            return {
                'title': candidate.strip(),
                'language': 'en'
            }

        return None

    def _build_wikipedia_url(self, title: str, language: str = "en") -> str:
        """
        Build Wikipedia URL from title and language.

        Args:
            title: Wikipedia page title
            language: Language code

        Returns:
            Full Wikipedia URL
        """
        # Replace spaces with underscores
        encoded_title = title.replace(' ', '_')
        return f"https://{language}.wikipedia.org/wiki/{encoded_title}"

    def _get_wikidata_id(self, wikipedia_title: str, language: str = "en") -> Optional[str]:
        """
        Get Wikidata ID from Wikipedia title using the Wikidata API.

        Args:
            wikipedia_title: Wikipedia page title
            language: Language code

        Returns:
            Wikidata ID (e.g., "Q1748") or None
        """
        try:
            # Check cache first (use separate cache for Wikidata IDs)
            cache_key = f"wikidata_{language}_{wikipedia_title}"
            if self.cache is not None:
                cached_qid = self.cache.get(cache_key)
                if cached_qid is not None:
                    return cached_qid

            # Normalize title (remove underscores)
            title = wikipedia_title.replace('_', ' ')

            # Query Wikidata API
            # Using the wbgetentities action with site and title
            url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'wbgetentities',
                'sites': f'{language}wiki',
                'titles': title,
                'format': 'json',
                'props': ''  # We only need the entity ID, not properties
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract Wikidata ID from response
            entities = data.get('entities', {})
            for entity_id, entity_data in entities.items():
                # Skip missing entities
                if entity_id == '-1' or 'missing' in entity_data:
                    continue

                # Found a valid Wikidata ID
                wikidata_id = entity_id

                # Cache the result
                if self.cache is not None:
                    self.cache.set(cache_key, wikidata_id)

                return wikidata_id

            # No valid entity found - cache None to avoid repeated API calls
            if self.cache is not None:
                self.cache.set(cache_key, None)

            return None

        except Exception as e:
            logger.debug(f"Error fetching Wikidata ID for '{wikipedia_title}': {e}")
            return None

    def _extract_language_variants(self, candidates: List[Dict]) -> Dict[str, str]:
        """
        Extract language variants from candidate list.

        Args:
            candidates: List of candidate dictionaries

        Returns:
            Dictionary mapping language codes to entity names
        """
        variants = {}
        for candidate in candidates:
            lang = candidate.get('language', 'en')
            title = candidate.get('wikipedia_title', '')
            if lang and title and lang not in variants:
                # Use the title without underscores as the variant
                variants[lang] = title.replace('_', ' ')
        return variants

    def _lookup_custom_kb(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """
        Look up entity in custom knowledge base.

        Args:
            entity_text: Entity text
            entity_type: Entity type

        Returns:
            Linking result dict or None
        """
        normalized_text = entity_text.lower()
        if normalized_text in self.custom_kb:
            kb_entry = self.custom_kb[normalized_text]

            # Verify type matches if specified
            if 'type' in kb_entry and kb_entry['type'] != entity_type:
                return None

            return {
                'wikipedia_title': kb_entry.get('canonical_name', entity_text),
                'wikidata_id': kb_entry.get('wikidata_id'),
                'wikipedia_url': kb_entry.get('wikipedia_url', ''),
                'canonical_name': kb_entry.get('canonical_name', entity_text),
                'language_variants': kb_entry.get('language_variants', {}),
                'linking_confidence': 1.0,  # Perfect match from KB
                'candidates': [],
                'disambiguation_method': 'custom_kb'
            }
        return None

    def _advanced_disambiguation(
        self,
        entity_text: str,
        candidates: List[Dict],
        context: str,
        co_entities: Optional[List[str]] = None
    ) -> Dict:
        """
        Advanced disambiguation using context embeddings and co-occurrence.

        Phase 3 Feature: Uses semantic similarity between context and candidate
        descriptions, plus entity co-occurrence patterns.

        Args:
            entity_text: Entity text
            candidates: List of candidate entities
            context: Context sentence
            co_entities: Other entities mentioned in context

        Returns:
            Best candidate with adjusted confidence
        """
        if not self.embedder or not candidates:
            return candidates[0] if candidates else None

        try:
            # Get context embedding
            context_embedding = self.embedder.encode(context, convert_to_tensor=True)

            # Score candidates based on context similarity
            best_candidate = candidates[0]
            best_score = best_candidate['confidence']

            for candidate in candidates[:3]:  # Top 3 candidates
                # Create candidate description
                title = candidate['wikipedia_title'].replace('_', ' ')
                candidate_desc = f"{title} ({candidate['language']})"

                # Compute semantic similarity
                cand_embedding = self.embedder.encode(candidate_desc, convert_to_tensor=True)
                similarity = torch.cosine_similarity(
                    context_embedding.unsqueeze(0),
                    cand_embedding.unsqueeze(0)
                ).item()

                # Adjust confidence based on similarity
                adjusted_confidence = (candidate['confidence'] + similarity) / 2.0

                # Bonus for co-occurrence patterns
                if co_entities and hasattr(self, 'entity_cooccurrence'):
                    # Check if this candidate co-occurs with other entities
                    wikidata_id = self._get_wikidata_id(candidate['wikipedia_title'], candidate['language'])
                    if wikidata_id and wikidata_id in self.entity_cooccurrence:
                        cooccur_count = sum(
                            self.entity_cooccurrence[wikidata_id][e]
                            for e in co_entities if e != entity_text
                        )
                        # Small boost for known co-occurrences
                        if cooccur_count > 0:
                            adjusted_confidence += min(0.1, cooccur_count * 0.02)

                # Update best if better
                if adjusted_confidence > best_score:
                    best_score = adjusted_confidence
                    best_candidate = candidate.copy()
                    best_candidate['confidence'] = adjusted_confidence

            return best_candidate

        except Exception as e:
            logger.warning(f"Advanced disambiguation failed: {e}")
            return candidates[0] if candidates else None

    def extract_entity_relationships(
        self,
        entities: List[Dict],
        text: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Extract relationships between entities mentioned in the same text.

        Phase 3 Feature: Identifies co-mentioned entities and their relationships.

        Args:
            entities: List of linked entities
            text: Source text containing entities
            relationship_types: Types of relationships to extract (default: ['co-mention'])

        Returns:
            List of relationship dictionaries:
            {
                'source_entity': str (wikidata_id),
                'target_entity': str (wikidata_id),
                'relationship_type': str,
                'confidence': float,
                'context': str
            }
        """
        if relationship_types is None:
            relationship_types = ['co-mention']

        relationships = []
        linked_entities = [e for e in entities if e.get('is_linked') and e.get('wikidata_id')]

        # Extract co-mention relationships
        if 'co-mention' in relationship_types and len(linked_entities) >= 2:
            for i, entity1 in enumerate(linked_entities):
                for entity2 in linked_entities[i+1:]:
                    relationships.append({
                        'source_entity': entity1['wikidata_id'],
                        'source_name': entity1.get('canonical_name', entity1['text']),
                        'target_entity': entity2['wikidata_id'],
                        'target_name': entity2.get('canonical_name', entity2['text']),
                        'relationship_type': 'co-mention',
                        'confidence': min(
                            entity1.get('linking_confidence', 1.0),
                            entity2.get('linking_confidence', 1.0)
                        ),
                        'context': text[:200]  # First 200 chars as context
                    })

        return relationships

    def get_entity_network(
        self,
        min_cooccurrence: int = 2
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get entity co-occurrence network for visualization.

        Phase 3 Feature: Returns network of entities that frequently co-occur.

        Args:
            min_cooccurrence: Minimum co-occurrence count

        Returns:
            Dictionary mapping entity IDs to list of (connected_entity, weight) tuples
        """
        network = {}
        for entity_id, cooccurrences in self.entity_cooccurrence.items():
            edges = [
                (co_entity, count)
                for co_entity, count in cooccurrences.items()
                if count >= min_cooccurrence
            ]
            if edges:
                network[entity_id] = edges
        return network

    def save_cooccurrence_data(self, output_path: str) -> None:
        """
        Save entity co-occurrence data for analysis.

        Phase 3 Feature: Exports co-occurrence patterns.

        Args:
            output_path: Path to save JSON file
        """
        data = {
            entity_id: dict(cooccur)
            for entity_id, cooccur in self.entity_cooccurrence.items()
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved co-occurrence data to {output_path}")

    def _get_entity_description(self, wikidata_id: str, language: str = "en") -> Optional[str]:
        """
        Retrieve entity description from Wikidata.

        Phase 4 Feature: Fetches human-readable descriptions for entities.

        Args:
            wikidata_id: Wikidata QID (e.g., "Q90")
            language: Language code for description

        Returns:
            Entity description or None
        """
        if not wikidata_id or not wikidata_id.startswith('Q'):
            return None

        # Check cache first
        cache_key = f"{wikidata_id}_{language}"
        if cache_key in self.entity_descriptions_cache:
            return self.entity_descriptions_cache[cache_key]

        try:
            # Query Wikidata API for entity description
            url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'wbgetentities',
                'ids': wikidata_id,
                'props': 'descriptions',
                'languages': language,
                'format': 'json'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract description
            entities = data.get('entities', {})
            if wikidata_id in entities:
                descriptions = entities[wikidata_id].get('descriptions', {})
                if language in descriptions:
                    description = descriptions[language].get('value', '')
                    # Cache the result
                    self.entity_descriptions_cache[cache_key] = description
                    return description

            # No description found - cache None
            self.entity_descriptions_cache[cache_key] = None
            return None

        except Exception as e:
            logger.debug(f"Error fetching description for {wikidata_id}: {e}")
            return None

    def set_document_context(self, document_text: str) -> None:
        """
        Set document-level context for disambiguation.

        Phase 4 Feature: Allows using entire document context instead of
        just sentence-level context.

        Args:
            document_text: Full document text
        """
        self.document_context = document_text
        logger.info(f"Document context set ({len(document_text)} chars)")

    def clear_document_context(self) -> None:
        """Clear document-level context."""
        self.document_context = None

    def extract_typed_relationships(
        self,
        entities: List[Dict],
        text: str
    ) -> List[Dict]:
        """
        Extract typed relationships between entities.

        Phase 4 Feature: Identifies specific relationship types based on
        entity types and textual patterns.

        Args:
            entities: List of linked entities with types
            text: Source text containing entities

        Returns:
            List of typed relationship dictionaries:
            {
                'source_entity': str (wikidata_id),
                'target_entity': str (wikidata_id),
                'relationship_type': str,
                'confidence': float,
                'evidence': str
            }
        """
        relationships = []
        linked_entities = [e for e in entities if e.get('is_linked') and e.get('wikidata_id')]

        if len(linked_entities) < 2:
            return relationships

        text_lower = text.lower()

        # Extract relationships based on entity type patterns
        for i, entity1 in enumerate(linked_entities):
            for entity2 in linked_entities[i+1:]:
                type1 = entity1.get('type', '')
                type2 = entity2.get('type', '')
                name1 = entity1.get('text', '')
                name2 = entity2.get('text', '')

                # Pattern: PER works-for ORG
                if type1 == 'PER' and type2 == 'ORG':
                    if self._check_works_for_pattern(name1, name2, text_lower):
                        relationships.append({
                            'source_entity': entity1['wikidata_id'],
                            'source_name': entity1.get('canonical_name', name1),
                            'target_entity': entity2['wikidata_id'],
                            'target_name': entity2.get('canonical_name', name2),
                            'relationship_type': 'works_for',
                            'confidence': 0.8,
                            'evidence': self._extract_evidence(name1, name2, text)
                        })

                # Pattern: LOC located-in LOC
                elif type1 == 'LOC' and type2 == 'LOC':
                    if self._check_located_in_pattern(name1, name2, text_lower):
                        relationships.append({
                            'source_entity': entity1['wikidata_id'],
                            'source_name': entity1.get('canonical_name', name1),
                            'target_entity': entity2['wikidata_id'],
                            'target_name': entity2.get('canonical_name', name2),
                            'relationship_type': 'located_in',
                            'confidence': 0.75,
                            'evidence': self._extract_evidence(name1, name2, text)
                        })

                # Pattern: ORG part-of ORG
                elif type1 == 'ORG' and type2 == 'ORG':
                    if self._check_part_of_pattern(name1, name2, text_lower):
                        relationships.append({
                            'source_entity': entity1['wikidata_id'],
                            'source_name': entity1.get('canonical_name', name1),
                            'target_entity': entity2['wikidata_id'],
                            'target_name': entity2.get('canonical_name', name2),
                            'relationship_type': 'part_of',
                            'confidence': 0.7,
                            'evidence': self._extract_evidence(name1, name2, text)
                        })

        return relationships

    def _check_works_for_pattern(self, person: str, org: str, text: str) -> bool:
        """Check if text contains works-for relationship pattern."""
        patterns = [
            f"{person.lower()}.{{0,30}}(works? for|employed by|ceo of|president of|director of)",
            f"(ceo|president|director|employee).{{0,30}}{person.lower()}.{{0,30}}{org.lower()}",
            f"{person.lower()}.{{0,30}}(at|@).{{0,10}}{org.lower()}"
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_located_in_pattern(self, loc1: str, loc2: str, text: str) -> bool:
        """Check if text contains located-in relationship pattern."""
        patterns = [
            f"{loc1.lower()}.{{0,30}}(in|located in|part of).{{0,10}}{loc2.lower()}",
            f"{loc2.lower()}.{{0,50}}{loc1.lower()}",  # Broader location mentioned first
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_part_of_pattern(self, org1: str, org2: str, text: str) -> bool:
        """Check if text contains part-of relationship pattern."""
        patterns = [
            f"{org1.lower()}.{{0,30}}(part of|subsidiary of|division of|owned by).{{0,10}}{org2.lower()}",
            f"{org2.lower()}'s.{{0,10}}{org1.lower()}"
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_evidence(self, entity1: str, entity2: str, text: str, max_len: int = 150) -> str:
        """Extract text snippet as evidence for relationship."""
        # Find the positions of both entities
        text_lower = text.lower()
        pos1 = text_lower.find(entity1.lower())
        pos2 = text_lower.find(entity2.lower())

        if pos1 == -1 or pos2 == -1:
            return text[:max_len]

        # Extract context around both mentions
        start = max(0, min(pos1, pos2) - 30)
        end = min(len(text), max(pos1, pos2) + max(len(entity1), len(entity2)) + 30)

        return text[start:end].strip()

    def clear_cache(self) -> None:
        """Clear all cached entity linking results."""
        if self.cache is not None:
            self.cache.clear()
            print("🗑️  Entity linking cache cleared")

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
    # Initialize linker
    linker = EntityLinker(enable_cache=True)

    # Single entity linking
    print("\n=== Single Entity Linking ===")
    result = linker.link_entity("Copenhagen", entity_type="LOC", language="en")
    if result:
        print(f"Entity: Copenhagen")
        print(f"Wikipedia: {result['wikipedia_url']}")
        print(f"Canonical: {result['canonical_name']}")
        print(f"Confidence: {result['linking_confidence']:.2f}")
        print(f"Variants: {result['language_variants']}")

    # Batch entity linking
    print("\n=== Batch Entity Linking ===")
    entities = [
        {'text': 'København', 'type': 'LOC', 'score': 0.95, 'language': 'da'},
        {'text': 'Mette Frederiksen', 'type': 'PER', 'score': 0.92, 'language': 'da'},
        {'text': 'Paris', 'type': 'LOC', 'score': 0.98, 'language': 'en'},
    ]

    enhanced = linker.link_entities_batch(entities)
    for ent in enhanced:
        print(f"\nOriginal: {ent['text']} ({ent.get('language', 'unknown')})")
        if ent.get('is_linked'):
            print(f"  → Linked to: {ent.get('canonical_name')}")
            print(f"  → Wikipedia: {ent.get('wikipedia_url')}")
            print(f"  → Confidence: {ent.get('linking_confidence', 0):.2f}")
        else:
            print(f"  → Not linked")

    # Cache stats
    print(f"\n=== Cache Stats ===")
    print(f"Cache: {linker.get_cache_stats()}")
