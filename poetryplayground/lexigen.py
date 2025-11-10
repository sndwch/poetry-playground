import random
from typing import List, Optional, TypeVar

import pronouncing

from .cache import cached_api_call
from .datamuse_api import get_datamuse_instance
from .utils import (
    sort_by_rarity,
    too_similar,
    validate_str_or_list_of_str,
    validate_word,
)
from .word_validator import word_validator


def clean_api_results(word_list, exclude_words=None, use_validator=True):
    """Clean and validate words from API results.

    Args:
        word_list: List of words to clean
        exclude_words: Words to exclude
        use_validator: Whether to use enhanced validation

    Returns:
        List of valid, cleaned words
    """
    if exclude_words is None:
        exclude_words = []

    # Use centralized WordValidator for all validation
    if use_validator:
        filtered = word_validator.clean_word_list(word_list, allow_rare=False, exclude_words=exclude_words)
    else:
        # Even without validator, filter out excluded words
        filtered = [w for w in word_list if w.lower() not in {e.lower() for e in exclude_words}]

    return filtered


def score_and_filter_results(
    api_response: List[dict],
    exclude_words: Optional[List[str]] = None,
    min_quality: float = 0.5,
    context: Optional['GenerationContext'] = None,
    use_datamuse_score: bool = True,
    sample_size: Optional[int] = None
) -> List[str]:
    """Score and filter API results using comprehensive quality metrics.

    Args:
        api_response: List of dicts from Datamuse API with 'word' and 'score' keys
        exclude_words: Words to exclude from results
        min_quality: Minimum quality threshold (0-1)
        context: Optional GenerationContext for tone/formality filtering
        use_datamuse_score: Whether to incorporate Datamuse relevance scores
        sample_size: If provided, return top N results after scoring

    Returns:
        List of words sorted by quality (best first)
    """
    from .quality_scorer import get_quality_scorer

    if not api_response:
        return []

    if exclude_words is None:
        exclude_words = []

    scorer = get_quality_scorer()
    exclude_set = {w.lower() for w in exclude_words}

    # Find max Datamuse score for normalization
    max_datamuse_score = max((item.get('score', 0) for item in api_response), default=1)
    if max_datamuse_score == 0:
        max_datamuse_score = 1  # Avoid division by zero

    # Score each word
    scored_words = []
    for item in api_response:
        word = item.get('word', '')
        if not word or word.lower() in exclude_set:
            continue

        # Validate word
        if not word_validator.is_valid_english_word(word, allow_rare=False):
            continue

        # Get Datamuse relevance score (normalized to 0-1)
        datamuse_score = 0.0
        if use_datamuse_score:
            raw_score = item.get('score', 0)
            datamuse_score = raw_score / max_datamuse_score if max_datamuse_score > 0 else 0.0

        # Get comprehensive quality score
        quality_score = scorer.score_word(word, context=context)

        # Combine scores: Datamuse relevance (30%) + Quality (70%)
        if use_datamuse_score:
            combined_score = (datamuse_score * 0.3) + (quality_score.overall * 0.7)
        else:
            combined_score = quality_score.overall

        # Filter by minimum quality
        if combined_score >= min_quality:
            scored_words.append((word, combined_score))

    # Sort by combined score (descending)
    scored_words.sort(key=lambda x: x[1], reverse=True)

    # Extract words (drop scores)
    words = [w for w, _ in scored_words]

    # Return sample if requested
    if sample_size and len(words) > sample_size:
        return words[:sample_size]

    return words


# Centralized Datamuse API instance (singleton)
api = get_datamuse_instance()
str_or_list_of_str = TypeVar("str_or_list_of_str", str, List[str])


# Cached API wrapper functions with 24-hour TTL for stability
@cached_api_call(endpoint="cmu.pronouncing.rhymes", ttl=86400)
def _cached_pronouncing_rhymes(word: str) -> List[str]:
    """Cached wrapper for CMU pronouncing dictionary rhyme lookups."""
    return pronouncing.rhymes(word)


@cached_api_call(endpoint="datamuse.similar_sounding", ttl=86400)
def _cached_datamuse_similar_sounding(word: str, max_results: int) -> List[dict]:
    """Cached wrapper for Datamuse similar sounding (sl) API calls."""
    return api.words(sl=word, max=max_results)


@cached_api_call(endpoint="datamuse.similar_meaning", ttl=86400)
def _cached_datamuse_similar_meaning(word: str, max_results: int) -> List[dict]:
    """Cached wrapper for Datamuse similar meaning (ml) API calls."""
    return api.words(ml=word, max=max_results)


@cached_api_call(endpoint="datamuse.contextually_linked", ttl=86400)
def _cached_datamuse_contextually_linked(word: str, max_results: int) -> List[dict]:
    """Cached wrapper for Datamuse contextually linked (rel_trg) API calls."""
    return api.words(rel_trg=word, max=max_results)


@cached_api_call(endpoint="datamuse.frequently_following", ttl=86400)
def _cached_datamuse_frequently_following(word: str, max_results: Optional[int]) -> List[dict]:
    """Cached wrapper for Datamuse frequently following (lc) API calls."""
    if max_results:
        return api.words(lc=word, max=max_results)
    return api.words(lc=word)


def rhymes(input_val: str_or_list_of_str, sample_size=None) -> List[str]:
    """Return a list of rhymes in randomized order for a given word if at least one can be found using the pronouncing
    module (which uses the CMU rhyming dictionary).

    :param input_val: the word or words in relation to which this function is looking up rhymes
    :param sample size: If provided, return a random sample of this many elements. If this number is greater than
                        the length of the rhyme list, then just return a shuffled copy of the rhyme list.
    """
    input_words = validate_str_or_list_of_str(input_val)
    rhyme_words: List[str] = []
    for input_word in input_words:
        # Use cached wrapper for CMU lookups
        cached_rhymes = _cached_pronouncing_rhymes(input_word)
        rhyme_words.extend(word_validator.clean_word_list(list(set(cached_rhymes)), allow_rare=False))
    return extract_sample(rhyme_words, sample_size=sample_size)


def rhyme(input_word: str) -> Optional[str]:
    """Return a random rhyme for a given word if at least one can be found using the pronouncing module (which uses
    the CMU rhyming dictionary).

    :param input_word: the word which this function is looking up a rhyme of
    """
    rhyme_list = rhymes(input_word)
    if len(rhyme_list):
        return next(iter(rhyme_list), None)
    return None


def extract_sample(word_list: list, sample_size: Optional[int] = None) -> list:
    """Returns a random sample from the word list or a shuffled copy of the word list.

    :param word_list: the list of words to extract the random sample from
    :param sample_size: If this number is greater than the length of the word list, then just return a shuffled
                        copy of the word list.
    """
    if not sample_size or len(word_list) <= sample_size:
        return random.sample(word_list, k=len(word_list))
    else:
        sample: List[str] = []
        while len(sample) < sample_size and len(word_list) > 0:
            sample += [
                word for word in random.sample(word_list, k=sample_size) if word not in sample
            ]
            word_list = [word for word in word_list if word not in sample]
        if sample_size < len(sample):
            return random.sample(sample, k=sample_size)
        return sample


def similar_sounding_words(
    input_val: str_or_list_of_str,
    sample_size: Optional[int] = 6,
    datamuse_api_max: Optional[int] = 50,
) -> list:
    """Return a list of similar sounding words to a given word, in randomized order, if at least one can be found using
    Datamuse API.

    :param input_val: the word or words in relation to which this function is looking up similar sounding words
    :param sample_size: If provided, return a random sample of this many elements. If this number is greater than the
                        length of the API results, then just return a shuffled copy of the filtered API results.
    :param datamuse_api_max: specifies the maximum number of results returned by the API. The API client's results
                             are always sorted from most to least similar sounding (according to a numeric score
                             provided by Datamuse), hence by using both parameters, one can control the size of both
                             the sample pool and the sample size.
    """
    input_words = validate_str_or_list_of_str(input_val)
    ss_words: List[str] = []
    for input_word in input_words:
        # Use cached wrapper for Datamuse API
        response = _cached_datamuse_similar_sounding(input_word, datamuse_api_max or 50)
        exclude_words = input_words + ss_words
        ss_words.extend(
            clean_api_results([obj["word"] for obj in response], exclude_words=exclude_words)
        )
    return extract_sample(ss_words, sample_size=sample_size)


def similar_sounding_word(input_word: str, datamuse_api_max: Optional[int] = 20) -> Optional[str]:
    """Return a random similar sounding word for a given word if at least one can be found using the Datamuse API.

    :param input_word: the word which this function is looking up a similar sounding words of
    :param datamuse_api_max: specifies the maximum number of results returned by the API. The API client's results are
                             always sorted from most to least similar sounding (according to a numeric score provided
                             by Datamuse).
    """
    return next(
        iter(similar_sounding_words(input_word, sample_size=1, datamuse_api_max=datamuse_api_max)),
        None,
    )


def similar_meaning_words(
    input_val: str_or_list_of_str,
    sample_size: Optional[int] = 6,
    datamuse_api_max: Optional[int] = 20,
    min_quality: float = 0.5,
    context: Optional['GenerationContext'] = None,
) -> list:
    """Return quality-filtered similar meaning words, sorted by quality.

    Uses comprehensive quality scoring to filter and rank results.

    :param input_val: the word or words in relation to which this function is looking up similar meaning words
    :param sample_size: If provided, return top N quality-ranked results
    :param datamuse_api_max: specifies the maximum number of results returned by the API
    :param min_quality: minimum quality threshold (0-1, default 0.5)
    :param context: optional GenerationContext for tone/formality filtering
    """
    input_words = validate_str_or_list_of_str(input_val)
    sm_words: List[str] = []
    for input_word in input_words:
        # Use cached wrapper for Datamuse API
        response = _cached_datamuse_similar_meaning(input_word, datamuse_api_max or 20)
        exclude_words = input_words + sm_words
        # Use quality-aware filtering
        quality_words = score_and_filter_results(
            response,
            exclude_words=exclude_words,
            min_quality=min_quality,
            context=context,
            use_datamuse_score=True,
            sample_size=sample_size
        )
        sm_words.extend(quality_words)

    # Already sorted by quality and sampled
    return sm_words[:sample_size] if sample_size else sm_words


def similar_meaning_word(input_word: str, datamuse_api_max: Optional[int] = 10) -> Optional[str]:
    """Return a random similar meaning word for a given word if at least one can be found using the Datamuse API.

    :param input_word: the word which this function is looking up a similar meaning words of
    :param datamuse_api_max: specifies the maximum number of results returned by the API. The API client's results are
                             always sorted from most to least similar meaning (according to a numeric score provided
                             by Datamuse).
    """
    return next(
        iter(similar_meaning_words(input_word, sample_size=1, datamuse_api_max=datamuse_api_max)),
        None,
    )


def contextually_linked_words(
    input_val: str_or_list_of_str,
    sample_size: Optional[int] = 6,
    datamuse_api_max: Optional[int] = 20,
    min_quality: float = 0.5,
    context: Optional['GenerationContext'] = None,
) -> list:
    """Return quality-filtered contextually linked words (collocations).

    Uses comprehensive quality scoring to avoid clichéd collocations.

    :param input_val: the word or words in relation to which this function is looking up contextually linked words
    :param sample_size: If provided, return top N quality-ranked results
    :param datamuse_api_max: specifies the maximum number of results returned by the API
    :param min_quality: minimum quality threshold (0-1, default 0.5)
    :param context: optional GenerationContext for tone/formality filtering
    """
    input_words = validate_str_or_list_of_str(input_val)
    cl_words: List[str] = []
    for input_word in input_words:
        validate_word(input_word)
        # Use cached wrapper for Datamuse API
        response = _cached_datamuse_contextually_linked(input_word, datamuse_api_max or 20)
        exclude_words = input_words + cl_words
        # Use quality-aware filtering with cliché detection
        quality_words = score_and_filter_results(
            response,
            exclude_words=exclude_words,
            min_quality=min_quality,
            context=context,
            use_datamuse_score=True,
            sample_size=sample_size
        )
        cl_words.extend(quality_words)

    # Already sorted by quality and sampled
    return cl_words[:sample_size] if sample_size else cl_words


def contextually_linked_word(
    input_word: str, datamuse_api_max: Optional[int] = 10
) -> Optional[str]:
    """Return a random word that frequently appear within the same document as a given word if at least one can be found
    using the Datamuse API.

    :param input_word: the word which this function is looking up a contextually linked words to
    :param datamuse_api_max: specifies the maximum number of results returned by the API. The API client's results are
                             always sorted from most to least similar sounding (according to a numeric score provided
                             by Datamuse).
    """
    return next(
        iter(
            contextually_linked_words(input_word, sample_size=1, datamuse_api_max=datamuse_api_max)
        ),
        None,
    )


def frequently_following_words(
    input_val: str_or_list_of_str,
    sample_size: Optional[int] = 8,
    datamuse_api_max: Optional[int] = None,
    min_quality: float = 0.5,
    context: Optional['GenerationContext'] = None,
) -> list:
    """Return quality-filtered words that frequently follow the given word.

    Combines quality scoring with rarity logic for diverse results.

    :param input_val: the word or words in relation to which this function is looking up frequently following words
    :param sample_size: If provided, return top N results (mix of quality + rare)
    :param datamuse_api_max: specifies the maximum number of results returned by the API
    :param min_quality: minimum quality threshold (0-1, default 0.5)
    :param context: optional GenerationContext for tone/formality filtering
    """
    input_words = validate_str_or_list_of_str(input_val)
    ff_words: List[str] = []
    for input_word in input_words:
        # Use cached wrapper for Datamuse API
        response = _cached_datamuse_frequently_following(input_word, datamuse_api_max)
        exclude_words = input_words + ff_words
        # Use quality-aware filtering
        quality_words = score_and_filter_results(
            response,
            exclude_words=exclude_words,
            min_quality=min_quality,
            context=context,
            use_datamuse_score=True,
            sample_size=None  # Get all quality words for rarity mixing
        )
        ff_words.extend(quality_words)

    # Mix quality with rarity for diversity
    if sample_size and sample_size > 4 and len(ff_words) > sample_size:
        # Take top quality words + some rare words for variety
        quality_count = sample_size - 3
        rare_count = 3

        # Top quality words (already sorted)
        quality_sample = ff_words[:quality_count]

        # Rare words from the quality-filtered set
        rare_candidates = sort_by_rarity(ff_words)[:20]
        rare_sample = extract_sample(rare_candidates, sample_size=rare_count)

        return quality_sample + rare_sample

    # Standard: return top quality words
    return ff_words[:sample_size] if sample_size else ff_words


def frequently_following_word(input_word, datamuse_api_max=10) -> Optional[str]:
    """Return a random word that frequently follows the given word if at least one can be found using the Datamuse API.

    :param input_word: the word which this function is looking up a frequently following word of
    :param datamuse_api_max: specifies the maximum number of results returned by the API. The API client's results are
                             always sorted from most to least similar sounding (according to a numeric score provided
                             by Datamuse).
    """
    result: Optional[str] = next(
        iter(
            frequently_following_words(input_word, sample_size=1, datamuse_api_max=datamuse_api_max)
        ),
        None,
    )
    return result


def phonetically_related_words(
    input_val: str_or_list_of_str,
    sample_size=None,
    datamuse_api_max=50,
    max_results_per_input_word: Optional[int] = None,
    min_quality: float = 0.5,
    context: Optional['GenerationContext'] = None,
) -> list:
    """Returns quality-filtered rhymes and similar sounding words.

    Uses comprehensive quality scoring to prefer high-quality phonetic matches.

    :param input_val: the word or words in relation to which this function is looking up phonetically related words
    :param sample_size: If provided, return top N quality-ranked results
    :param datamuse_api_max: specifies how many API results can be returned by the API client
    :param max_results_per_input_word: limit the number of output words per input word
    :param min_quality: minimum quality threshold (0-1, default 0.5)
    :param context: optional GenerationContext for tone/formality filtering
    """
    from .quality_scorer import get_quality_scorer

    input_words = validate_str_or_list_of_str(input_val)
    scorer = get_quality_scorer()
    results: List[str] = []

    for word in input_words:
        # Get rhymes and similar sounding words
        rhyme_words = rhymes(word, sample_size=max_results_per_input_word)
        exclude_words = input_words + results + rhyme_words
        similar_words = similar_sounding_words(
            word, sample_size=sample_size, datamuse_api_max=datamuse_api_max
        )

        # Combine and filter
        all_phonetic = rhyme_words + [w for w in similar_words if w not in exclude_words]

        # Apply quality scoring to phonetic matches
        scored_phonetic = []
        for phonetic_word in all_phonetic:
            if word_validator.is_valid_english_word(phonetic_word, allow_rare=False):
                quality_score = scorer.score_word(phonetic_word, context=context)
                if quality_score.overall >= min_quality:
                    scored_phonetic.append((phonetic_word, quality_score.overall))

        # Sort by quality
        scored_phonetic.sort(key=lambda x: x[1], reverse=True)

        # Take top results per input word
        if max_results_per_input_word:
            results.extend([w for w, _ in scored_phonetic[:max_results_per_input_word]])
        else:
            results.extend([w for w, _ in scored_phonetic])

    # Return top quality results
    return results[:sample_size] if sample_size else results


def related_rare_words(
    input_val: str_or_list_of_str,
    sample_size: Optional[int] = 8,
    rare_word_population_max: int = 20,
    min_quality: float = 0.5,
    context: Optional['GenerationContext'] = None,
) -> list:
    """Return quality-filtered rare related words (phonetic, contextual, or semantic).

    Combines rarity with quality scoring to ensure rare words are also poetically valuable.
    Words must pass quality thresholds while being sorted by rarity.

    :param input_val: the word or words in relation to which this function is looking up related rare words
    :param sample_size: If provided, return a random sample of this many elements. If this number is greater than
                        the length of rare word population size, then just return a shuffled copy of that.
    :param rare_word_population_max: specifies the maximum number of related words to subsample from per word.
                                     The rare word population is sorted from rarest to most common. If sample_size is
                                     null, the max results returned by this function is 2 times this number.
    :param min_quality: minimum quality score (0-1) for words to be included (default 0.5)
    :param context: optional GenerationContext for tone/formality filtering
    """
    input_words = validate_str_or_list_of_str(input_val)
    results: List[str] = []
    for input_word in input_words:
        # Get quality-filtered phonetically related words
        related_words = phonetically_related_words(
            input_word,
            sample_size=None,
            min_quality=min_quality,
            context=context
        )

        # Add quality-filtered contextually linked words
        related_words.extend(
            word
            for word in contextually_linked_words(
                input_word,
                sample_size=None,
                datamuse_api_max=100,
                min_quality=min_quality,
                context=context
            )
            if word not in related_words
        )

        # Add quality-filtered similar meaning words
        related_words.extend(
            word
            for word in similar_meaning_words(
                input_word,
                sample_size=None,
                datamuse_api_max=100,
                min_quality=min_quality,
                context=context
            )
            if word not in related_words
        )

        # Filter out words too similar to input
        related_words = [word for word in related_words if not too_similar(input_word, word)]

        # Sort by rarity and take the rarest words (already quality-filtered)
        results.extend(sort_by_rarity(related_words)[:rare_word_population_max])

    return extract_sample(results, sample_size=sample_size)


def related_rare_word(
    input_word: str,
    rare_word_population_max: int = 10,
    min_quality: float = 0.5,
    context: Optional['GenerationContext'] = None,
) -> Optional[str]:
    """Return a quality-filtered random rare related word (phonetic, contextual, or semantic).

    :param input_word: the word which this function is looking up related rare words to
    :param rare_word_population_max: specifies the maximum number of related words to subsample from. The rare word
                                    population is sorted from rarest to most common.
    :param min_quality: minimum quality score (0-1) for words to be included (default 0.5)
    :param context: optional GenerationContext for tone/formality filtering
    """
    return next(
        iter(
            related_rare_words(
                input_word,
                sample_size=1,
                rare_word_population_max=rare_word_population_max,
                min_quality=min_quality,
                context=context
            )
        ),
        None,
    )
