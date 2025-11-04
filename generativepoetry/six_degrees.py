"""
Six Degrees - Bidirectional Word Convergence

Explores word relationships from two starting points until they converge,
inspired by the "Six Degrees of Wikipedia" phenomenon. Discovers the
conceptual pathways between seemingly unrelated words.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datamuse import datamuse

from .word_validator import WordValidator


@dataclass
class WordNode:
    """A word in the exploration tree"""
    word: str
    level: int
    parent: Optional['WordNode'] = None
    relationship_type: str = ""
    confidence: float = 0.0


@dataclass
class ConvergencePath:
    """A complete path from word A to word B through convergence"""
    word_a: str
    word_b: str
    convergence_word: str
    path_a: List[WordNode]  # Path from A to convergence
    path_b: List[WordNode]  # Path from B to convergence
    total_steps: int
    relationship_chain: List[str]

    def get_full_path(self) -> List[str]:
        """Get the complete word path from A to B"""
        # Reverse path_a since we built it from convergence back to A
        forward_path = [node.word for node in reversed(self.path_a)]
        # Add convergence word
        forward_path.append(self.convergence_word)
        # Add path_b (already in correct direction)
        forward_path.extend([node.word for node in self.path_b])
        return forward_path


class SixDegrees:
    """Bidirectional word exploration to find convergence paths"""

    def __init__(self, max_levels: int = 3, max_words_per_level: int = 8):
        self.datamuse_api = datamuse.Datamuse()
        self.word_validator = WordValidator()
        self.max_levels = max_levels
        self.max_words_per_level = max_words_per_level

        # Streamlined relationship types (focus on most effective ones)
        self.relationship_types = [
            ('ml', 'similar_meaning', 1.0),      # means like
            ('rel_trg', 'contextual', 0.8),     # frequent followers
            ('rel_syn', 'synonym', 0.9),        # synonyms
            ('rel_gen', 'general', 0.7),        # more general terms
        ]

    def find_convergence(self, word_a: str, word_b: str) -> Optional[ConvergencePath]:
        """Find convergence path between two words"""
        print(f"🔍 Exploring convergence between '{word_a}' and '{word_b}'...")

        # Initialize exploration trees
        tree_a = {0: [WordNode(word_a, 0)]}  # level -> nodes
        tree_b = {0: [WordNode(word_b, 0)]}

        # Track all words seen from each direction
        words_from_a = {word_a: WordNode(word_a, 0)}
        words_from_b = {word_b: WordNode(word_b, 0)}

        # Explore level by level
        for level in range(1, self.max_levels + 1):
            print(f"  Level {level}...")

            # Expand from A
            if level - 1 in tree_a:
                new_nodes_a = self._expand_level(tree_a[level - 1], level, 'A')
                tree_a[level] = new_nodes_a
                for node in new_nodes_a:
                    words_from_a[node.word] = node

            # Expand from B
            if level - 1 in tree_b:
                new_nodes_b = self._expand_level(tree_b[level - 1], level, 'B')
                tree_b[level] = new_nodes_b
                for node in new_nodes_b:
                    words_from_b[node.word] = node

            # Check for convergence
            convergence = self._check_convergence(words_from_a, words_from_b)
            if convergence:
                convergence_word, node_a, node_b = convergence
                print(f"  ✅ Convergence found at '{convergence_word}' (level {level})")

                # Reconstruct paths
                path_a = self._reconstruct_path(node_a)
                path_b = self._reconstruct_path(node_b)

                return ConvergencePath(
                    word_a=word_a,
                    word_b=word_b,
                    convergence_word=convergence_word,
                    path_a=path_a,
                    path_b=path_b,
                    total_steps=len(path_a) + len(path_b),
                    relationship_chain=self._extract_relationship_chain(path_a, path_b)
                )

            # Brief pause to avoid API rate limits
            time.sleep(0.5)

        print(f"  ❌ No convergence found within {self.max_levels} levels")
        return None

    def _expand_level(self, parent_nodes: List[WordNode], level: int, direction: str) -> List[WordNode]:
        """Expand one level from parent nodes"""
        new_nodes = []

        for parent in parent_nodes:
            # Get related words for this parent
            related_words = self._get_related_words(parent.word)

            # Limit to prevent explosion
            if len(related_words) > self.max_words_per_level:
                # Sort by confidence and take top ones
                related_words.sort(key=lambda x: x[2], reverse=True)
                related_words = related_words[:self.max_words_per_level]

            for word, rel_type, confidence in related_words:
                if self._is_valid_expansion_word(word):
                    node = WordNode(
                        word=word,
                        level=level,
                        parent=parent,
                        relationship_type=rel_type,
                        confidence=confidence
                    )
                    new_nodes.append(node)

        print(f"    {direction}: Found {len(new_nodes)} new words")
        return new_nodes

    def _get_related_words(self, word: str) -> List[Tuple[str, str, float]]:
        """Get related words using multiple relationship types"""
        all_related = []

        for rel_code, rel_name, base_confidence in self.relationship_types:
            try:
                # Query Datamuse API
                if rel_code == 'ml':
                    results = self.datamuse_api.words(ml=word, max=10)
                elif rel_code.startswith('rel_'):
                    query = {rel_code: word, 'max': 8}
                    results = self.datamuse_api.words(**query)
                else:
                    continue

                # Process results
                for result in results:
                    if 'word' in result:
                        word_result = result['word']

                        # Skip multi-word phrases and compounds for now
                        if ' ' in word_result or '-' in word_result:
                            continue

                        # Skip if same as input word
                        if word_result.lower() == word.lower():
                            continue

                        # Adjust confidence based on API score and our base confidence
                        api_score = result.get('score', 1000) / 1000.0  # Normalize
                        final_confidence = min(1.0, api_score * base_confidence)

                        all_related.append((word_result, rel_name, final_confidence))

                # Small delay between API calls
                time.sleep(0.2)

            except Exception as e:
                print(f"    Warning: Error getting {rel_name} for '{word}': {e}")
                continue

        return all_related

    def _is_valid_expansion_word(self, word: str) -> bool:
        """Check if word is suitable for expansion"""
        if not word or len(word) < 2:
            return False

        if not word.isalpha():
            return False

        # Skip only the most common words that might create false convergences
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'it', 'for', 'not', 'on', 'with', 'as', 'do', 'at', 'this',
            'but', 'by', 'from', 'they', 'we', 'or', 'an', 'will'
        }

        if word.lower() in common_words:
            return False

        return self.word_validator.is_valid_english_word(word)

    def _check_convergence(self, words_from_a: Dict[str, WordNode],
                          words_from_b: Dict[str, WordNode]) -> Optional[Tuple[str, WordNode, WordNode]]:
        """Check if any words from A and B have converged"""
        common_words = set(words_from_a.keys()) & set(words_from_b.keys())

        if common_words:
            # If multiple convergences, pick the one with shortest total path
            best_convergence = None
            best_total_distance = float('inf')

            for word in common_words:
                node_a = words_from_a[word]
                node_b = words_from_b[word]
                total_distance = node_a.level + node_b.level

                if total_distance < best_total_distance:
                    best_total_distance = total_distance
                    best_convergence = (word, node_a, node_b)

            return best_convergence

        return None

    def _reconstruct_path(self, node: WordNode) -> List[WordNode]:
        """Reconstruct path from root to this node"""
        path = []
        current = node
        while current.parent is not None:
            path.append(current)
            current = current.parent
        return path

    def _extract_relationship_chain(self, path_a: List[WordNode], path_b: List[WordNode]) -> List[str]:
        """Extract the chain of relationships used"""
        chain = []

        # Add relationships from A side (reversed)
        for node in reversed(path_a):
            if node.relationship_type:
                chain.append(node.relationship_type)

        # Add relationships from B side
        for node in path_b:
            if node.relationship_type:
                chain.append(node.relationship_type)

        return chain

    def format_convergence_report(self, convergence: ConvergencePath) -> str:
        """Format a nice report of the convergence path"""
        if not convergence:
            return "No convergence path found."

        report = []
        report.append("🔗 SIX DEGREES CONVERGENCE FOUND")
        report.append("=" * 50)

        # Show the complete path
        full_path = convergence.get_full_path()
        report.append(f"\n📍 Path: {' → '.join(full_path)}")
        report.append(f"📏 Total steps: {convergence.total_steps}")
        report.append(f"🎯 Convergence point: '{convergence.convergence_word}'")

        # Show detailed paths
        report.append(f"\n🔵 From '{convergence.word_a}':")
        current = convergence.word_a
        for node in reversed(convergence.path_a):
            report.append(f"  {current} --({node.relationship_type})--> {node.word}")
            current = node.word
        if convergence.path_a:
            report.append(f"  {current} ---> {convergence.convergence_word}")

        report.append(f"\n🔴 From '{convergence.word_b}':")
        current = convergence.word_b
        for node in reversed(convergence.path_b):
            report.append(f"  {current} --({node.relationship_type})--> {node.word}")
            current = node.word
        if convergence.path_b:
            report.append(f"  {current} ---> {convergence.convergence_word}")

        # Show relationship types used
        if convergence.relationship_chain:
            report.append(f"\n🔧 Relationship types: {', '.join(set(convergence.relationship_chain))}")

        return "\n".join(report)

    def calculate_semantic_distance(self, word_a: str, word_b: str) -> Dict[str, Any]:
        """Calculate semantic distance between two words"""
        print(f"🔢 Calculating semantic distance between '{word_a}' and '{word_b}'...")

        # Try to find convergence with extended search
        original_max_levels = self.max_levels
        self.max_levels = 4  # Deeper search for distance calculation

        convergence = self.find_convergence(word_a, word_b)

        if convergence:
            # Found convergence - distance is total steps
            distance_score = convergence.total_steps
            relationship_strength = 1.0 / (distance_score + 1)  # Closer = stronger
            distance_category = "close" if distance_score <= 3 else "moderate" if distance_score <= 6 else "distant"
        else:
            # No convergence - they're very distant
            distance_score = float('inf')
            relationship_strength = 0.0
            distance_category = "very_distant"

        # Restore original max levels
        self.max_levels = original_max_levels

        return {
            'word_a': word_a,
            'word_b': word_b,
            'distance_score': distance_score,
            'relationship_strength': relationship_strength,
            'category': distance_category,
            'convergence_path': convergence,
            'has_semantic_bridge': convergence is not None
        }

    def explore_multiple_paths(self, word_a: str, word_b: str, num_attempts: int = 3) -> List[ConvergencePath]:
        """Try to find multiple different convergence paths"""
        paths = []

        for attempt in range(num_attempts):
            print(f"\n--- Attempt {attempt + 1} ---")
            path = self.find_convergence(word_a, word_b)
            if path:
                paths.append(path)

            # Brief pause between attempts
            time.sleep(1)

        return paths
