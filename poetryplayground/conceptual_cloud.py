"""Conceptual cloud generator - poet's radar for word associations.

This module unifies the existing lexical tools to generate six types of word
clusters around a center word/phrase:
    1. Semantic: near-meaning words (spaCy embeddings + Datamuse)
    2. Contextual: words that appear WITH the center (collocation)
    3. Opposite: antonyms and polar opposites
    4. Phonetic: rhymes, near-rhymes, alliteration
    5. Imagery: concrete nouns with sensory qualities
    6. Rare: unusual but related words ("strange orbit")

Example:
    >>> cloud = generate_conceptual_cloud("fire")
    >>> print(format_as_rich(cloud))
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from poetryplayground.cache import cached_api_call
from poetryplayground.datamuse_api import DatamuseAPI
from poetryplayground.lexicon import get_lexicon_data
from poetryplayground.lexigen import (
    contextually_linked_words,
    frequently_following_words,
    phonetically_related_words,
    related_rare_words,
    similar_meaning_words,
)
from poetryplayground.logger import logger
from poetryplayground.word_validator import WordValidator

# ============================================================================
# Enums and Constants
# ============================================================================


class ClusterType(str, Enum):
    """Types of word clusters."""

    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"
    OPPOSITE = "opposite"
    PHONETIC = "phonetic"
    IMAGERY = "imagery"
    RARE = "rare"


CLUSTER_DESCRIPTIONS = {
    ClusterType.SEMANTIC: "Near-meaning words (synonyms, related concepts)",
    ClusterType.CONTEXTUAL: "Words that appear together (collocation)",
    ClusterType.OPPOSITE: "Antonyms and polar opposites",
    ClusterType.PHONETIC: "Rhymes, near-rhymes, similar sounds",
    ClusterType.IMAGERY: "Concrete nouns with sensory qualities",
    ClusterType.RARE: "Unusual but related words (strange orbit)",
}


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CloudTerm:
    """A single term in a conceptual cloud cluster.

    Attributes:
        term: The word itself
        cluster_type: Which cluster it belongs to
        score: Relevance/similarity score (0.0-1.0)
        freq_bucket: Frequency category ('common', 'mid', 'rare')
        metadata: Optional additional info (POS, syllables, etc.)
    """

    term: str
    cluster_type: ClusterType
    score: float
    freq_bucket: str = "mid"
    metadata: Dict[str, any] = field(default_factory=dict)

    def __str__(self):
        return f"{self.term} ({self.score:.2f})"


@dataclass
class ConceptualCloud:
    """Complete conceptual cloud with all clusters.

    Attributes:
        center_word: The center word/phrase
        clusters: Dict mapping ClusterType to list of CloudTerms
        total_terms: Total number of terms across all clusters
        config: Configuration used to generate this cloud
        timestamp: When this cloud was generated
    """

    center_word: str
    clusters: Dict[ClusterType, List[CloudTerm]]
    total_terms: int = 0
    config: Optional["CloudConfig"] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Calculate total terms after initialization."""
        self.total_terms = sum(len(terms) for terms in self.clusters.values())

        if self.timestamp is None:
            from datetime import datetime

            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_cluster(self, cluster_type: ClusterType) -> List[CloudTerm]:
        """Get terms for a specific cluster type."""
        return self.clusters.get(cluster_type, [])

    def get_all_terms(self) -> List[str]:
        """Get all terms as a flat list."""
        all_terms = []
        for terms in self.clusters.values():
            all_terms.extend(t.term for t in terms)
        return all_terms


@dataclass
class CloudConfig:
    """Configuration for cloud generation.

    Attributes:
        k_per_cluster: Number of terms per cluster
        total_limit: Maximum total terms across all clusters
        sections: Which cluster types to include (None = all)
        include_scores: Whether to include similarity scores
        cache_results: Whether to cache API calls
        min_score: Minimum similarity score threshold
        allow_rare: Allow rare/obscure words
    """

    k_per_cluster: int = 10
    total_limit: int = 50
    sections: Optional[List[ClusterType]] = None
    include_scores: bool = True
    cache_results: bool = True
    min_score: float = 0.0
    allow_rare: bool = True

    def __post_init__(self):
        """Set default sections if not provided."""
        if self.sections is None:
            self.sections = list(ClusterType)


# ============================================================================
# Helper Functions - Antonyms (NEW)
# ============================================================================


@cached_api_call(endpoint="datamuse.antonyms", ttl=86400)
def _cached_datamuse_antonyms(word: str, max_results: int) -> List[Dict[str, any]]:
    """Cached Datamuse antonym lookup.

    Args:
        word: Word to find antonyms for
        max_results: Maximum number of results

    Returns:
        List of dicts with 'word' and 'score' keys
    """
    api = DatamuseAPI()
    results = api.words(rel_ant=word, max=max_results)
    return results


def get_antonyms(word: str, k: int = 10) -> List[Tuple[str, float]]:
    """Get antonyms for a word using Datamuse.

    Args:
        word: Word to find antonyms for
        k: Number of antonyms to return

    Returns:
        List of (antonym, score) tuples with scores normalized to 0-1
    """
    try:
        results = _cached_datamuse_antonyms(word, max_results=k * 2)

        if not results:
            return []

        # Find max score for normalization
        max_score = max(item.get("score", 0) for item in results)
        if max_score == 0:
            max_score = 1.0  # Avoid division by zero

        # Validate results
        validator = WordValidator()
        valid_results = []

        for item in results:
            ant_word = item.get("word", "")
            raw_score = item.get("score", 0)
            # Normalize to 0-1 based on max score in this result set
            score = raw_score / max_score if max_score > 0 else 0.0

            if validator.is_valid_english_word(ant_word):
                valid_results.append((ant_word, score))

        return valid_results[:k]

    except Exception as e:
        logger.warning(f"Failed to get antonyms for '{word}': {e}")
        return []


# ============================================================================
# Helper Functions - Imagery/Concreteness
# ============================================================================


def get_concrete_nouns(word_list: List[str], k: int = 10) -> List[str]:
    """Filter word list for concrete nouns.

    Uses POS tagging to find nouns, with preference for concrete/sensory words.

    Args:
        word_list: List of words to filter
        k: Number of nouns to return

    Returns:
        List of concrete noun words
    """
    lexicon = get_lexicon_data()
    concrete_nouns = []

    for word in word_list:
        # Check if it's a noun
        pos = lexicon.pos_cache.get(word.lower())
        if pos and pos == "NOUN":
            concrete_nouns.append(word)

    return concrete_nouns[:k]


def get_frequency_bucket(word: str) -> str:
    """Categorize word by frequency.

    Args:
        word: Word to categorize

    Returns:
        'common', 'mid', or 'rare'
    """
    lexicon = get_lexicon_data()
    zipf = lexicon.zipf_cache.get(word.lower(), 0.0)

    # Zipf scale: 1-10, higher = more common
    if zipf >= 5.0:
        return "common"
    elif zipf >= 3.0:
        return "mid"
    else:
        return "rare"


# ============================================================================
# Cluster Generators
# ============================================================================


def _generate_semantic_cluster(word: str, k: int) -> List[CloudTerm]:
    """Generate semantic/near-meaning cluster.

    Uses spaCy embeddings + Datamuse means-like endpoint with quality scoring.

    Args:
        word: Center word
        k: Number of terms to generate

    Returns:
        List of CloudTerms with real quality scores
    """
    from poetryplayground.quality_scorer import get_quality_scorer

    terms = []
    scorer = get_quality_scorer()

    try:
        # Get similar words using existing function
        similar_words = similar_meaning_words(word, sample_size=k * 2)

        # Also try semantic space if available
        try:
            from poetryplayground.semantic_geodesic import get_semantic_space

            space = get_semantic_space()
            vec = space.get_vector(word)
            neighbors = space.find_nearest(vec, k=k, exclude={word})

            # Merge results
            for neighbor_word, _score in neighbors:
                if neighbor_word not in similar_words:
                    similar_words.append(neighbor_word)
        except Exception as e:
            logger.debug(f"Semantic space not available: {e}")

        # Score each word and sort by quality
        scored_words = []
        for w in similar_words[: k * 2]:  # Get extras for filtering
            quality_score = scorer.score_word(w)
            scored_words.append((w, quality_score.overall))

        # Sort by quality and take top k
        scored_words.sort(key=lambda x: x[1], reverse=True)

        # Convert to CloudTerms
        for w, quality in scored_words[:k]:
            terms.append(
                CloudTerm(
                    term=w,
                    cluster_type=ClusterType.SEMANTIC,
                    score=quality,  # Real quality score
                    freq_bucket=get_frequency_bucket(w),
                )
            )

    except Exception as e:
        logger.warning(f"Failed to generate semantic cluster: {e}")

    return terms


def _generate_contextual_cluster(word: str, k: int) -> List[CloudTerm]:
    """Generate contextual/collocation cluster.

    Uses Datamuse rel_trg + lc (left context) endpoints with quality scoring.

    Args:
        word: Center word
        k: Number of terms to generate

    Returns:
        List of CloudTerms with real quality scores
    """
    from poetryplayground.quality_scorer import get_quality_scorer

    terms = []
    scorer = get_quality_scorer()

    try:
        # Get contextually linked words
        ctx_words = contextually_linked_words(word, sample_size=k)

        # Get frequently following words
        follow_words = frequently_following_words(word, sample_size=k)

        # Combine and deduplicate
        all_words = list(set(ctx_words + follow_words))

        # Score each word and sort by quality
        scored_words = []
        for w in all_words[: k * 2]:  # Get extras for filtering
            quality_score = scorer.score_word(w)
            # Also check if this creates a clichéd phrase with the center word
            phrase_penalty = 0.0
            test_phrase = f"{word} {w}"
            if scorer.is_cliche(test_phrase, threshold=0.6):
                phrase_penalty = 0.2
            scored_words.append((w, quality_score.overall - phrase_penalty))

        # Sort by quality and take top k
        scored_words.sort(key=lambda x: x[1], reverse=True)

        # Convert to CloudTerms
        for w, quality in scored_words[:k]:
            terms.append(
                CloudTerm(
                    term=w,
                    cluster_type=ClusterType.CONTEXTUAL,
                    score=max(0.0, quality),  # Real quality score (clamped to 0)
                    freq_bucket=get_frequency_bucket(w),
                )
            )

    except Exception as e:
        logger.warning(f"Failed to generate contextual cluster: {e}")

    return terms


def _generate_opposite_cluster(word: str, k: int) -> List[CloudTerm]:
    """Generate opposite/antonym cluster.

    Uses Datamuse rel_ant endpoint with quality scoring.

    Args:
        word: Center word
        k: Number of terms to generate

    Returns:
        List of CloudTerms with real quality scores
    """
    from poetryplayground.quality_scorer import get_quality_scorer

    terms = []
    scorer = get_quality_scorer()

    try:
        antonyms = get_antonyms(word, k=k * 2)

        # Score each antonym for quality
        scored_words = []
        for ant_word, datamuse_score in antonyms:
            quality_score = scorer.score_word(ant_word)
            # Combine Datamuse relevance (normalized) with quality
            # Datamuse scores are already 0-1 normalized in get_antonyms()
            combined = (quality_score.overall * 0.7) + (datamuse_score * 0.3)
            scored_words.append((ant_word, combined))

        # Sort by combined score
        scored_words.sort(key=lambda x: x[1], reverse=True)

        # Convert to CloudTerms
        for ant_word, quality in scored_words[:k]:
            terms.append(
                CloudTerm(
                    term=ant_word,
                    cluster_type=ClusterType.OPPOSITE,
                    score=quality,  # Combined quality + Datamuse relevance
                    freq_bucket=get_frequency_bucket(ant_word),
                )
            )

    except Exception as e:
        logger.warning(f"Failed to generate opposite cluster: {e}")

    return terms


def _generate_phonetic_cluster(word: str, k: int) -> List[CloudTerm]:
    """Generate phonetic/rhyme cluster.

    Uses CMU Pronouncing Dict + Datamuse sounds-like endpoint with quality scoring.

    Args:
        word: Center word
        k: Number of terms to generate

    Returns:
        List of CloudTerms with real quality scores
    """
    from poetryplayground.quality_scorer import get_quality_scorer

    terms = []
    scorer = get_quality_scorer()

    try:
        # Get phonetically related words (rhymes + sounds-like)
        phon_words = phonetically_related_words(word, sample_size=k * 2)

        # Score and sort by quality
        scored_words = []
        for w in phon_words:
            quality_score = scorer.score_word(w)
            scored_words.append((w, quality_score.overall))

        # Sort by quality
        scored_words.sort(key=lambda x: x[1], reverse=True)

        # Convert to CloudTerms
        for w, quality in scored_words[:k]:
            terms.append(
                CloudTerm(
                    term=w,
                    cluster_type=ClusterType.PHONETIC,
                    score=quality,  # Real quality score
                    freq_bucket=get_frequency_bucket(w),
                )
            )

    except Exception as e:
        logger.warning(f"Failed to generate phonetic cluster: {e}")

    return terms


def _generate_imagery_cluster(word: str, k: int) -> List[CloudTerm]:
    """Generate imagery/concrete noun cluster.

    Gets semantic neighbors, then filters for concrete nouns with quality scoring.

    Args:
        word: Center word
        k: Number of terms to generate

    Returns:
        List of CloudTerms with real quality scores
    """
    from poetryplayground.quality_scorer import get_quality_scorer

    terms = []
    scorer = get_quality_scorer()

    try:
        # Start with semantic neighbors
        semantic_words = similar_meaning_words(word, sample_size=k * 3)

        # Filter for concrete nouns
        concrete = get_concrete_nouns(semantic_words, k=k * 2)

        # Score by quality, prioritizing high concreteness
        scored_words = []
        for w in concrete:
            quality_score = scorer.score_word(w)
            concreteness = scorer.get_concreteness(w)
            # Combined score: favor both quality and concreteness
            combined = (quality_score.overall * 0.5) + (concreteness * 0.5)
            scored_words.append((w, combined, concreteness))

        # Sort by combined score
        scored_words.sort(key=lambda x: x[1], reverse=True)

        # Convert to CloudTerms
        for w, combined, concreteness in scored_words[:k]:
            terms.append(
                CloudTerm(
                    term=w,
                    cluster_type=ClusterType.IMAGERY,
                    score=combined,  # Combined quality + concreteness score
                    freq_bucket=get_frequency_bucket(w),
                    metadata={"pos": "NOUN", "concreteness": concreteness},
                )
            )

    except Exception as e:
        logger.warning(f"Failed to generate imagery cluster: {e}")

    return terms


def _generate_rare_cluster(word: str, k: int) -> List[CloudTerm]:
    """Generate rare but related cluster ("strange orbit").

    Uses existing related_rare_words function with quality scoring.

    Args:
        word: Center word
        k: Number of terms to generate

    Returns:
        List of CloudTerms with real quality scores
    """
    from poetryplayground.quality_scorer import get_quality_scorer

    terms = []
    scorer = get_quality_scorer()

    try:
        # Get rare but related words
        rare_words = related_rare_words(word, sample_size=k * 2)

        # Score each rare word for quality
        # For rare words, we want high quality but also true rarity
        scored_words = []
        for w in rare_words:
            quality_score = scorer.score_word(w)
            # Bonus for being genuinely rare (not just low quality)
            freq_bucket = get_frequency_bucket(w)
            rarity_bonus = 0.1 if freq_bucket == "rare" else 0.0
            combined = quality_score.overall + rarity_bonus
            scored_words.append((w, combined))

        # Sort by combined score
        scored_words.sort(key=lambda x: x[1], reverse=True)

        # Convert to CloudTerms
        for w, quality in scored_words[:k]:
            terms.append(
                CloudTerm(
                    term=w,
                    cluster_type=ClusterType.RARE,
                    score=min(1.0, quality),  # Real quality score with rarity bonus
                    freq_bucket=get_frequency_bucket(w),
                )
            )

    except Exception as e:
        logger.warning(f"Failed to generate rare cluster: {e}")

    return terms


# ============================================================================
# Main Orchestrator
# ============================================================================


def generate_conceptual_cloud(
    center_word: str,
    k_per_cluster: int = 10,
    total_limit: int = 50,
    sections: Optional[List[str]] = None,
    include_scores: bool = True,
    cache_results: bool = True,
) -> ConceptualCloud:
    """Generate a complete conceptual cloud around a center word.

    This is the main entry point that orchestrates all cluster generators.

    Args:
        center_word: The word or phrase to center the cloud on
        k_per_cluster: Number of terms per cluster (default: 10)
        total_limit: Maximum total terms (default: 50)
        sections: List of section names to include (None = all 6)
        include_scores: Whether to include similarity scores
        cache_results: Whether to use cached API results

    Returns:
        ConceptualCloud with all requested clusters

    Example:
        >>> cloud = generate_conceptual_cloud("fire", k_per_cluster=5)
        >>> print(f"Generated {cloud.total_terms} terms")
        >>> for cluster_type, terms in cloud.clusters.items():
        ...     print(f"{cluster_type}: {[t.term for t in terms]}")
    """
    # Parse sections
    if sections is None:
        sections_enum = list(ClusterType)
    else:
        sections_enum = []
        for s in sections:
            try:
                if isinstance(s, ClusterType):
                    sections_enum.append(s)
                else:
                    sections_enum.append(ClusterType(s))
            except ValueError:
                logger.warning(f"Unknown section type: {s}")

    # Create config
    config = CloudConfig(
        k_per_cluster=k_per_cluster,
        total_limit=total_limit,
        sections=sections_enum,
        include_scores=include_scores,
        cache_results=cache_results,
    )

    logger.info(
        f"Generating conceptual cloud for '{center_word}' with {len(sections_enum)} sections"
    )

    # Generate clusters
    clusters = {}

    for cluster_type in sections_enum:
        logger.debug(f"Generating {cluster_type.value} cluster...")

        if cluster_type == ClusterType.SEMANTIC:
            clusters[cluster_type] = _generate_semantic_cluster(center_word, k_per_cluster)
        elif cluster_type == ClusterType.CONTEXTUAL:
            clusters[cluster_type] = _generate_contextual_cluster(center_word, k_per_cluster)
        elif cluster_type == ClusterType.OPPOSITE:
            clusters[cluster_type] = _generate_opposite_cluster(center_word, k_per_cluster)
        elif cluster_type == ClusterType.PHONETIC:
            clusters[cluster_type] = _generate_phonetic_cluster(center_word, k_per_cluster)
        elif cluster_type == ClusterType.IMAGERY:
            clusters[cluster_type] = _generate_imagery_cluster(center_word, k_per_cluster)
        elif cluster_type == ClusterType.RARE:
            clusters[cluster_type] = _generate_rare_cluster(center_word, k_per_cluster)

    # Create cloud
    cloud = ConceptualCloud(
        center_word=center_word,
        clusters=clusters,
        config=config,
    )

    logger.info(f"Generated cloud with {cloud.total_terms} total terms")

    return cloud


# ============================================================================
# Output Formatters
# ============================================================================


def format_as_rich(cloud: ConceptualCloud, show_scores: bool = True) -> str:
    """Format cloud as Rich table for terminal display.

    Args:
        cloud: ConceptualCloud to format
        show_scores: Whether to show similarity scores

    Returns:
        Rich-formatted string (will be rendered by Rich console)
    """
    # Create table
    table = Table(
        title=f"Conceptual Cloud: '{cloud.center_word}'",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    # Add columns for each cluster type
    active_clusters = [ct for ct in ClusterType if cloud.clusters.get(ct)]

    for cluster_type in active_clusters:
        table.add_column(
            cluster_type.value.capitalize(),
            style="cyan",
            no_wrap=True,
        )

    # Find max rows needed
    max_rows = max(len(cloud.clusters.get(ct, [])) for ct in active_clusters)

    # Add rows
    for i in range(max_rows):
        row = []
        for cluster_type in active_clusters:
            terms = cloud.clusters.get(cluster_type, [])
            if i < len(terms):
                term = terms[i]
                # Color by frequency
                if term.freq_bucket == "common":
                    color = "green"
                elif term.freq_bucket == "rare":
                    color = "magenta"
                else:
                    color = "yellow"

                if show_scores:
                    cell = f"[{color}]{term.term}[/{color}] ({term.score:.2f})"
                else:
                    cell = f"[{color}]{term.term}[/{color}]"
                row.append(cell)
            else:
                row.append("")
        table.add_row(*row)

    # Render to string
    from io import StringIO

    string_io = StringIO()
    # Use wider width for TUI panels (was 120, now 160 to use full space)
    temp_console = Console(file=string_io, force_terminal=True, width=160)
    temp_console.print(table)

    # Add metadata panel
    metadata = f"Total terms: {cloud.total_terms} | Generated: {cloud.timestamp}"
    temp_console.print(Panel(metadata, box=box.ROUNDED, style="dim"))

    return string_io.getvalue()


def format_as_json(cloud: ConceptualCloud) -> str:
    """Format cloud as JSON.

    Args:
        cloud: ConceptualCloud to format

    Returns:
        JSON string
    """
    data = {
        "center_word": cloud.center_word,
        "timestamp": cloud.timestamp,
        "total_terms": cloud.total_terms,
        "clusters": {},
    }

    for cluster_type, terms in cloud.clusters.items():
        data["clusters"][cluster_type.value] = [
            {
                "term": t.term,
                "score": t.score,
                "freq_bucket": t.freq_bucket,
                "metadata": t.metadata,
            }
            for t in terms
        ]

    return json.dumps(data, indent=2)


def format_as_markdown(cloud: ConceptualCloud, show_scores: bool = True) -> str:
    """Format cloud as Markdown.

    Args:
        cloud: ConceptualCloud to format
        show_scores: Whether to show similarity scores

    Returns:
        Markdown-formatted string
    """
    lines = [
        f"# Conceptual Cloud: {cloud.center_word}",
        "",
        f"*Generated: {cloud.timestamp} | Total terms: {cloud.total_terms}*",
        "",
    ]

    for cluster_type, terms in cloud.clusters.items():
        if not terms:
            continue

        lines.append(f"## {cluster_type.value.capitalize()}")
        lines.append("")
        lines.append(CLUSTER_DESCRIPTIONS[cluster_type])
        lines.append("")

        for term in terms:
            if show_scores:
                lines.append(f"- **{term.term}** ({term.score:.2f}) - *{term.freq_bucket}*")
            else:
                lines.append(f"- **{term.term}** - *{term.freq_bucket}*")

        lines.append("")

    return "\n".join(lines)


def format_as_simple(cloud: ConceptualCloud) -> str:
    """Format cloud as simple text (word lists).

    Args:
        cloud: ConceptualCloud to format

    Returns:
        Plain text string
    """
    lines = [
        f"Conceptual Cloud: {cloud.center_word}",
        f"Generated: {cloud.timestamp}",
        f"Total terms: {cloud.total_terms}",
        "",
    ]

    for cluster_type, terms in cloud.clusters.items():
        if not terms:
            continue

        lines.append(f"{cluster_type.value.upper()}:")
        lines.append("  " + ", ".join(t.term for t in terms))
        lines.append("")

    return "\n".join(lines)
