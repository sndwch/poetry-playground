"""Semantic geodesic/bridge finder for poetry ideation.

Finds semantic paths between words using spaCy word vectors and
sklearn k-NN indexing for fast nearest-neighbor queries.

This module implements the semantic geodesic algorithm that finds transitional
paths through meaning-space, enabling poets to explore gradual transformations
between concepts (e.g., fire → flame → heat → warmth → cool → frost → ice).

Example:
    >>> from poetryplayground.semantic_geodesic import find_semantic_path
    >>> path = find_semantic_path("hot", "cold", steps=5)
    >>> print(" → ".join(path.get_primary_path()))
    hot → warm → tepid → cool → cold
"""

import functools
import heapq
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
from sklearn.neighbors import NearestNeighbors

from poetryplayground.core.lexicon import LexiconData, get_lexicon_data
from poetryplayground.logger import logger
from poetryplayground.setup_models import lazy_ensure_spacy_model

# ============================================================================
# Dataclasses for Results
# ============================================================================


@dataclass
class BridgeWord:
    """A word along the semantic path.

    Attributes:
        word: The bridge word itself
        position: Position along the path (0.0 to 1.0)
        similarity: Cosine similarity to ideal position
        deviation: Distance from the ideal geodesic line
        syllables: Number of syllables (if available)
        pos: Part of speech tag (if available)
    """

    word: str
    position: float  # 0.0 to 1.0
    similarity: float
    deviation: float
    syllables: Optional[int] = None
    pos: Optional[str] = None

    def __str__(self):
        return f"{self.word} (t={self.position:.2f}, sim={self.similarity:.3f})"


@dataclass
class SemanticPath:
    """Complete semantic path from start to end.

    Attributes:
        start: Starting word
        end: Ending word
        bridges: List of lists of bridge words (k alternatives per step)
        method: Path method used ("linear", "bezier", "shortest")
        smoothness_score: Average similarity between adjacent words
        deviation_score: Average deviation from ideal path
        diversity_score: Variance in alternative candidates
    """

    start: str
    end: str
    bridges: List[List[BridgeWord]]  # k alternatives per step
    method: str = "linear"
    smoothness_score: float = 0.0
    deviation_score: float = 0.0
    diversity_score: float = 0.0

    def get_primary_path(self) -> List[str]:
        """Get single best path (top candidate at each step)."""
        if not self.bridges:
            return [self.start, self.end]
        return [self.start] + [step[0].word for step in self.bridges] + [self.end]

    def get_all_alternatives(self) -> List[List[str]]:
        """Get all alternative words at each step."""
        return [[bridge.word for bridge in step] for step in self.bridges]


# ============================================================================
# Semantic Space Management
# ============================================================================


def _get_model_description(model_name: str) -> str:
    """Get human-readable description for spaCy model."""
    model_descriptions = {
        "en_core_web_sm": "English language model (small, ~12MB)",
        "en_core_web_md": "English language model (medium, ~40MB)",
        "en_core_web_lg": "English language model (large, ~400MB)",
    }
    return model_descriptions.get(model_name, f"spaCy model '{model_name}'")


class SemanticSpace:
    """Manages word embeddings and k-NN indexing for fast queries.

    This class handles:
    - Automatically downloading spaCy models if not installed
    - Loading spaCy models efficiently (disabled unused components)
    - Building a k-NN index over the vocabulary for fast similarity search
    - Caching the index to disk to avoid rebuilds
    - Context-aware vector retrieval for polysemy handling

    Args:
        model_name: spaCy model to use (en_core_web_sm/md/lg)
        vocab_size: Number of words to index (default 50k)
        cache_dir: Directory for caching k-NN index

    Note:
        If the specified spaCy model is not installed, it will be
        automatically downloaded on first use. Large models (~400MB)
        may take a few minutes to download.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        vocab_size: int = 50000,
        cache_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.vocab_size = vocab_size

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "poetryplayground" / "semantic_space"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure model is installed (download if necessary)
        model_description = _get_model_description(model_name)
        if not lazy_ensure_spacy_model(model_name, model_description):
            raise RuntimeError(
                f"Failed to load or download spaCy model '{model_name}'. "
                f"Please install it manually: python -m spacy download {model_name}"
            )

        # Load spaCy model (disable unused components for speed)
        logger.info(f"Loading spaCy model: {model_name}...")
        self.nlp = spacy.load(
            model_name,
            disable=["parser", "ner", "lemmatizer"],  # Only need vectors
        )
        logger.info(f"Loaded {len(self.nlp.vocab)} words with vectors")

        # Build or load cached k-NN index
        self._build_or_load_index()

    def _build_or_load_index(self):
        """Build k-NN index or load from cache."""
        # Include version in cache key to invalidate on algorithm changes
        cache_version = "v3"  # Increment when changing k-NN config or filtering
        cache_key = f"knn_index_{self.model_name}_{self.vocab_size}_{cache_version}.pkl"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            logger.info(f"Loading cached k-NN index from {cache_path}...")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
                self.index_to_word = cached["index_to_word"]
                self.word_to_index = cached["word_to_index"]
                self.vectors = cached["vectors"]
                self.nn_index = cached["nn_index"]
            logger.info(f"Loaded index with {len(self.index_to_word)} words")
            return

        # Build new index
        logger.info(f"Building vocabulary index (top {self.vocab_size} words)...")

        # Get all words with vectors from the vocabulary
        # Note: Iterate through vocab and check for vectors
        valid_words = []
        for word_text in self.nlp.vocab.strings:
            try:
                lex = self.nlp.vocab[word_text]
                if not lex.has_vector:
                    continue

                text = lex.text.lower()

                # Filter out unwanted words
                # Note: Keeping filters minimal to work with various spaCy models
                if (
                    len(text) >= 3  # Minimum 3 letters
                    and text.isalpha()  # Use Python's isalpha for compatibility
                    and not any(c in text for c in "''-_")  # Skip contractions/hyphenated
                ):
                    valid_words.append(lex)
            except (KeyError, AttributeError):
                # Skip words that cause errors
                continue

        logger.info(f"Found {len(valid_words)} valid words with vectors")

        # Validate we have enough words
        # If no valid words found, this likely indicates a test environment issue
        # Create a minimal fallback to allow tests to continue
        if len(valid_words) == 0:
            logger.warning("No valid words found with vectors. Using minimal fallback vocabulary.")
            # Create a minimal vocab from common words that should exist
            fallback_words = ["the", "is", "and", "of", "to", "a", "in", "for", "on", "with"]
            for word in fallback_words:
                try:
                    lex = self.nlp.vocab[word]
                    if lex and lex.has_vector:
                        valid_words.append(lex)
                except (KeyError, AttributeError):
                    pass

            # If still no words, raise error
            if len(valid_words) == 0:
                raise RuntimeError(
                    "Insufficient vocabulary: unable to find any valid words with vectors. "
                    "This indicates a problem with the spaCy model installation."
                )

        # Sort by frequency (log probability), take top N
        valid_words.sort(key=lambda w: w.prob, reverse=True)
        valid_words = valid_words[: self.vocab_size]

        # Build index structures
        self.index_to_word = [word.text.lower() for word in valid_words]
        self.word_to_index = {word: i for i, word in enumerate(self.index_to_word)}
        self.vectors = np.array([word.vector for word in valid_words])

        # Build k-NN index
        # Note: Using 'brute' algorithm because 'cosine' metric is not supported by 'ball_tree'
        logger.info("Building k-NN index...")
        n_neighbors = min(100, max(1, len(self.vectors)))  # Ensure at least 1
        self.nn_index = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="brute", metric="cosine"
        )
        self.nn_index.fit(self.vectors)

        # Cache to disk
        logger.info(f"Caching k-NN index to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "index_to_word": self.index_to_word,
                    "word_to_index": self.word_to_index,
                    "vectors": self.vectors,
                    "nn_index": self.nn_index,
                },
                f,
            )

        logger.info(f"Indexed {len(self.index_to_word)} words")

    def get_vector(self, word: str, context: Optional[str] = None) -> np.ndarray:
        """Get embedding vector for a word, optionally with context.

        Args:
            word: The word to get vector for
            context: Optional context to disambiguate polysemous words
                    (e.g., get_vector("bank", "river") vs get_vector("bank", "money"))

        Returns:
            300-dimensional vector
        """
        if context:
            # Contextualize by using last word in phrase
            doc = self.nlp(f"{context} {word}")
            return doc[-1].vector
        return self.nlp(word).vector

    def find_nearest(
        self, target_vec: np.ndarray, k: int = 1, exclude: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """Find k nearest words to target vector.

        Args:
            target_vec: Target vector to find neighbors for
            k: Number of neighbors to return
            exclude: Words to exclude from results

        Returns:
            List of (word, similarity) tuples, sorted by similarity
        """
        # Fetch more than k to account for exclusions
        n_fetch = min(k * 3, len(self.vectors))
        distances, indices = self.nn_index.kneighbors([target_vec], n_neighbors=n_fetch)

        results = []
        exclude = exclude or set()

        for dist, idx in zip(distances[0], indices[0]):
            word = self.index_to_word[idx]
            if word not in exclude:
                # Convert distance to similarity (cosine)
                similarity = 1 - dist
                results.append((word, similarity))
                if len(results) >= k:
                    break

        return results

    def get_similarity(self, word1: str, word2: str) -> float:
        """Get cosine similarity between two words.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Cosine similarity (0 to 1)
        """
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        return dot_product / (norm1 * norm2)


# Singleton cached instance
@functools.lru_cache(maxsize=1)
def get_semantic_space(model_name: str = "en_core_web_lg") -> SemanticSpace:
    """Get or create cached semantic space.

    Args:
        model_name: spaCy model to use

    Returns:
        Cached SemanticSpace instance
    """
    return SemanticSpace(model_name)


# ============================================================================
# Path Finding Functions
# ============================================================================


def find_semantic_path(
    start: str,
    end: str,
    steps: int = 5,
    k: int = 1,
    min_zipf: float = 3.0,
    pos_filter: Optional[str] = None,
    syllable_min: Optional[int] = None,
    syllable_max: Optional[int] = None,
    method: str = "linear",
    control_words: Optional[List[str]] = None,
    semantic_space: Optional[SemanticSpace] = None,
    lexicon_data: Optional[LexiconData] = None,
) -> SemanticPath:
    """Find semantic path from start to end word.

    This is the main entry point for semantic path finding. It supports
    multiple path methods and various filtering options.

    Args:
        start: Starting word
        end: Ending word
        steps: Total steps including start/end (minimum 3)
        k: Number of alternative candidates per step
        min_zipf: Minimum word frequency (1-10 scale)
        pos_filter: Filter by POS (NOUN, VERB, ADJ, ADV)
        syllable_min: Minimum syllables
        syllable_max: Maximum syllables
        method: Path method ("linear", "bezier", "shortest")
        control_words: Control points for Bezier curves
        semantic_space: Pre-loaded semantic space (for testing)
        lexicon_data: Pre-loaded lexicon (for testing)

    Returns:
        SemanticPath with bridges and metadata

    Raises:
        ValueError: If steps < 3 or invalid method
    """
    # Validation
    if steps < 3:
        raise ValueError("Steps must be >= 3 (start, at least 1 bridge, end)")

    if method not in ("linear", "bezier", "shortest"):
        raise ValueError(f"Invalid method: {method}. Must be linear, bezier, or shortest")

    # Dispatch to method-specific function
    if method == "linear":
        return _find_linear_path(
            start,
            end,
            steps,
            k,
            min_zipf,
            pos_filter,
            syllable_min,
            syllable_max,
            semantic_space,
            lexicon_data,
        )
    elif method == "bezier":
        return _find_bezier_path(
            start,
            end,
            steps,
            k,
            min_zipf,
            pos_filter,
            syllable_min,
            syllable_max,
            control_words,
            semantic_space,
            lexicon_data,
        )
    else:  # shortest
        return _find_shortest_path(
            start,
            end,
            steps,
            k,
            min_zipf,
            pos_filter,
            syllable_min,
            syllable_max,
            semantic_space,
            lexicon_data,
        )


def _find_linear_path(
    start: str,
    end: str,
    steps: int,
    k: int,
    min_zipf: float,
    pos_filter: Optional[str],
    syllable_min: Optional[int],
    syllable_max: Optional[int],
    semantic_space: Optional[SemanticSpace],
    lexicon_data: Optional[LexiconData],
) -> SemanticPath:
    """Find linear interpolation path.

    This uses simple linear interpolation: vec(t) = (1-t)*vec_start + t*vec_end
    """
    # Load resources
    if semantic_space is None:
        semantic_space = get_semantic_space()
    if lexicon_data is None:
        lexicon_data = get_lexicon_data()

    # Get vectors
    vec_start = semantic_space.get_vector(start.lower())
    vec_end = semantic_space.get_vector(end.lower())
    path_vec = vec_end - vec_start

    # Find bridges
    bridges = []
    used_words = {start.lower(), end.lower()}

    for i in range(1, steps - 1):
        t = i / (steps - 1)
        vec_t = vec_start + t * path_vec

        # Find candidates
        candidates = semantic_space.find_nearest(vec_t, k=k * 5, exclude=used_words)

        # Apply filters
        filtered = []
        for word, sim in candidates:
            if not _passes_filters(
                word, min_zipf, pos_filter, syllable_min, syllable_max, lexicon_data
            ):
                continue

            # Calculate deviation from ideal line
            word_vec = semantic_space.get_vector(word)
            ideal_vec = vec_start + t * path_vec
            deviation = 1 - (
                np.dot(word_vec, ideal_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(ideal_vec))
            )

            # Create BridgeWord
            bridge = BridgeWord(
                word=word,
                position=t,
                similarity=sim,
                deviation=deviation,
                syllables=lexicon_data.syllable_cache.get(word),
                pos=lexicon_data.pos_cache.get(word),
            )
            filtered.append(bridge)
            used_words.add(word.lower())

            if len(filtered) >= k:
                break

        bridges.append(filtered)

    # Calculate quality metrics
    smoothness = _calculate_smoothness(start, end, bridges, semantic_space)
    deviation = _calculate_deviation(bridges)
    diversity = _calculate_diversity(bridges)

    return SemanticPath(
        start=start,
        end=end,
        bridges=bridges,
        method="linear",
        smoothness_score=smoothness,
        deviation_score=deviation,
        diversity_score=diversity,
    )


def _find_bezier_path(
    start: str,
    end: str,
    steps: int,
    k: int,
    min_zipf: float,
    pos_filter: Optional[str],
    syllable_min: Optional[int],
    syllable_max: Optional[int],
    control_words: Optional[List[str]],
    semantic_space: Optional[SemanticSpace],
    lexicon_data: Optional[LexiconData],
) -> SemanticPath:
    """Find Bezier curve path through semantic space.

    Uses cubic Bezier: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    """
    # Load resources
    if semantic_space is None:
        semantic_space = get_semantic_space()
    if lexicon_data is None:
        lexicon_data = get_lexicon_data()

    # Get control points
    vec_start = semantic_space.get_vector(start.lower())
    vec_end = semantic_space.get_vector(end.lower())

    # Auto-generate control points if not provided
    if control_words is None or len(control_words) < 2:
        # Use midpoint perpendicular offset
        midpoint = (vec_start + vec_end) / 2
        vec_p1 = vec_start + 0.3 * (midpoint - vec_start)
        vec_p2 = vec_end + 0.3 * (midpoint - vec_end)
    else:
        vec_p1 = semantic_space.get_vector(control_words[0].lower())
        vec_p2 = semantic_space.get_vector(
            control_words[1].lower() if len(control_words) > 1 else control_words[0].lower()
        )

    # Find bridges using Bezier interpolation
    bridges = []
    used_words = {start.lower(), end.lower()}
    if control_words:
        used_words.update(w.lower() for w in control_words)

    for i in range(1, steps - 1):
        t = i / (steps - 1)

        # Cubic Bezier formula
        vec_t = (
            (1 - t) ** 3 * vec_start
            + 3 * (1 - t) ** 2 * t * vec_p1
            + 3 * (1 - t) * t**2 * vec_p2
            + t**3 * vec_end
        )

        # Find and filter candidates (same as linear)
        candidates = semantic_space.find_nearest(vec_t, k=k * 5, exclude=used_words)

        filtered = []
        for word, sim in candidates:
            if not _passes_filters(
                word, min_zipf, pos_filter, syllable_min, syllable_max, lexicon_data
            ):
                continue

            word_vec = semantic_space.get_vector(word)
            deviation = 1 - (
                np.dot(word_vec, vec_t) / (np.linalg.norm(word_vec) * np.linalg.norm(vec_t))
            )

            bridge = BridgeWord(
                word=word,
                position=t,
                similarity=sim,
                deviation=deviation,
                syllables=lexicon_data.syllable_cache.get(word),
                pos=lexicon_data.pos_cache.get(word),
            )
            filtered.append(bridge)
            used_words.add(word.lower())

            if len(filtered) >= k:
                break

        bridges.append(filtered)

    # Calculate quality metrics
    smoothness = _calculate_smoothness(start, end, bridges, semantic_space)
    deviation = _calculate_deviation(bridges)
    diversity = _calculate_diversity(bridges)

    return SemanticPath(
        start=start,
        end=end,
        bridges=bridges,
        method="bezier",
        smoothness_score=smoothness,
        deviation_score=deviation,
        diversity_score=diversity,
    )


def _find_shortest_path(
    start: str,
    end: str,
    steps: int,
    k: int,
    min_zipf: float,
    pos_filter: Optional[str],
    syllable_min: Optional[int],
    syllable_max: Optional[int],
    semantic_space: Optional[SemanticSpace],
    lexicon_data: Optional[LexiconData],
) -> SemanticPath:
    """Find shortest path through similarity graph using A*.

    This builds a local graph around start and end words, then uses A*
    to find the shortest path through high-similarity connections.
    """
    # Load resources
    if semantic_space is None:
        semantic_space = get_semantic_space()
    if lexicon_data is None:
        lexicon_data = get_lexicon_data()

    # Build local graph (BFS from both ends)
    max_graph_size = min(1000, semantic_space.vocab_size // 10)
    graph = _build_local_graph(
        start,
        end,
        max_graph_size,
        semantic_space,
        lexicon_data,
        min_zipf,
        pos_filter,
        syllable_min,
        syllable_max,
    )

    # A* search
    path_words = _astar_search(start.lower(), end.lower(), graph, semantic_space)

    if path_words is None or len(path_words) < steps:
        # Fallback to linear if no path found or path too short
        logger.warning("Shortest path failed, falling back to linear")
        return _find_linear_path(
            start,
            end,
            steps,
            k,
            min_zipf,
            pos_filter,
            syllable_min,
            syllable_max,
            semantic_space,
            lexicon_data,
        )

    # Subsample to get desired number of steps
    bridge_words = _subsample_path(path_words, steps)

    # Convert to BridgeWord objects
    bridges = []
    for i, word in enumerate(bridge_words):
        t = (i + 1) / (steps - 1)
        sim = semantic_space.get_similarity(word, start if i == 0 else bridge_words[i - 1])

        bridge = BridgeWord(
            word=word,
            position=t,
            similarity=sim,
            deviation=0.0,  # Not applicable for graph paths
            syllables=lexicon_data.syllable_cache.get(word),
            pos=lexicon_data.pos_cache.get(word),
        )
        bridges.append([bridge])  # Single alternative per step

    # Calculate quality metrics
    smoothness = _calculate_smoothness(start, end, bridges, semantic_space)
    diversity = 0.0  # No alternatives in shortest path

    return SemanticPath(
        start=start,
        end=end,
        bridges=bridges,
        method="shortest",
        smoothness_score=smoothness,
        deviation_score=0.0,
        diversity_score=diversity,
    )


# ============================================================================
# Helper Functions
# ============================================================================


def _passes_filters(
    word: str,
    min_zipf: float,
    pos_filter: Optional[str],
    syllable_min: Optional[int],
    syllable_max: Optional[int],
    lexicon_data: LexiconData,
) -> bool:
    """Check if word passes all filters."""
    # Frequency filter
    if lexicon_data.zipf_cache.get(word, 0) < min_zipf:
        return False

    # POS filter
    if pos_filter:
        pos = lexicon_data.pos_cache.get(word)
        if pos != pos_filter:
            return False

    # Syllable filter
    if syllable_min is not None or syllable_max is not None:
        syl = lexicon_data.syllable_cache.get(word)
        if syl is None:
            return False
        if syllable_min is not None and syl < syllable_min:
            return False
        if syllable_max is not None and syl > syllable_max:
            return False

    return True


def _calculate_smoothness(
    start: str, end: str, bridges: List[List[BridgeWord]], semantic_space: SemanticSpace
) -> float:
    """Calculate smoothness score (avg similarity between adjacent words)."""
    if not bridges:
        return 0.0

    path = [start] + [step[0].word for step in bridges] + [end]
    similarities = []

    for i in range(len(path) - 1):
        sim = semantic_space.get_similarity(path[i], path[i + 1])
        similarities.append(sim)

    return float(np.mean(similarities))


def _calculate_deviation(bridges: List[List[BridgeWord]]) -> float:
    """Calculate average deviation from ideal path."""
    if not bridges:
        return 0.0

    deviations = [step[0].deviation for step in bridges if step]
    return float(np.mean(deviations)) if deviations else 0.0


def _calculate_diversity(bridges: List[List[BridgeWord]]) -> float:
    """Calculate diversity score (variance in alternatives)."""
    if not bridges or all(len(step) < 2 for step in bridges):
        return 0.0

    # Calculate variance in similarity scores within each step
    variances = []
    for step in bridges:
        if len(step) >= 2:
            sims = [b.similarity for b in step]
            variances.append(np.var(sims))

    return float(np.mean(variances)) if variances else 0.0


def _build_local_graph(
    start: str,
    end: str,
    max_size: int,
    semantic_space: SemanticSpace,
    lexicon_data: LexiconData,
    min_zipf: float,
    pos_filter: Optional[str],
    syllable_min: Optional[int],
    syllable_max: Optional[int],
) -> Dict[str, List[Tuple[str, float]]]:
    """Build local similarity graph around start and end words."""
    graph = {}
    visited = set()
    frontier = [(start.lower(),), (end.lower(),)]

    while frontier and len(visited) < max_size:
        word = frontier.pop(0)[0] if frontier else None
        if not word or word in visited:
            continue

        visited.add(word)

        # Find neighbors
        vec = semantic_space.get_vector(word)
        neighbors = semantic_space.find_nearest(vec, k=20, exclude=visited)

        # Filter neighbors
        filtered_neighbors = []
        for neighbor, sim in neighbors:
            if _passes_filters(
                neighbor, min_zipf, pos_filter, syllable_min, syllable_max, lexicon_data
            ):
                filtered_neighbors.append((neighbor, sim))
                if neighbor not in visited:
                    frontier.append((neighbor,))

        graph[word] = filtered_neighbors

    return graph


def _astar_search(
    start: str,
    end: str,
    graph: Dict[str, List[Tuple[str, float]]],
    semantic_space: SemanticSpace,
) -> Optional[List[str]]:
    """A* search for shortest path through similarity graph."""
    if start not in graph or end not in graph:
        return None

    # Heuristic: 1 - similarity to end
    def heuristic(word: str) -> float:
        return 1 - semantic_space.get_similarity(word, end)

    # Priority queue: (f_score, word, path)
    heap = [(heuristic(start), start, [start])]
    visited = set()

    while heap:
        f_score, current, path = heapq.heappop(heap)

        if current == end:
            return path

        if current in visited:
            continue
        visited.add(current)

        # Explore neighbors
        for neighbor, _sim in graph.get(current, []):
            if neighbor not in visited:
                new_path = [*path, neighbor]
                g_score = len(new_path)  # Path length
                h_score = heuristic(neighbor)
                f_score = g_score + h_score

                heapq.heappush(heap, (f_score, neighbor, new_path))

    return None  # No path found


def _subsample_path(path: List[str], target_steps: int) -> List[str]:
    """Subsample path to get target number of steps."""
    if len(path) <= target_steps:
        return path[1:-1]  # Exclude start and end

    # Evenly subsample
    indices = np.linspace(1, len(path) - 2, target_steps - 2, dtype=int)
    return [path[i] for i in indices]
