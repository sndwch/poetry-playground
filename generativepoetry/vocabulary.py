"""
Shared Vocabulary Lists

Centralized collection of evocative words, atmospheric nouns, emotional keywords,
and other vocabulary sets used across multiple poetry generation modules.
"""

from typing import Dict, Set, List


class SharedVocabulary:
    """Centralized vocabulary for poetry generation"""

    def __init__(self):
        self._evocative_verbs = None
        self._atmospheric_nouns = None
        self._emotional_keywords = None
        self._poetic_attributes = None

    @property
    def evocative_verbs(self) -> Set[str]:
        """Rich collection of evocative verbs for poetry"""
        if self._evocative_verbs is None:
            self._evocative_verbs = {
                'ache', 'alight', 'alloy', 'allude', 'anchor', 'angle', 'animate', 'anneal', 'answer', 'arc', 'arouse', 'arrange', 'arrive', 'ascend', 'ash', 'ask', 'aspire', 'atomize', 'atone', 'awaken', 'backscatter', 'bandage', 'banish', 'bare', 'bathe', 'beckon', 'befall', 'befriend', 'beg', 'beget', 'begin', 'behold', 'belie', 'belong', 'bend', 'bequeath', 'bereave', 'beseech', 'besiege', 'bestow', 'bleed', 'blend', 'blink', 'bloom', 'blossom', 'blot', 'blur', 'bode', 'bolster', 'bond', 'braise', 'braid', 'branch', 'brand', 'breathe', 'bridle', 'brighten', 'brim', 'bristle', 'brood', 'brown', 'bud', 'buffer', 'build', 'burn', 'burnish', 'burrow', 'burgeon', 'bury', 'bustle', 'cage', 'calve', 'cancel', 'candle', 'capsize', 'capture', 'careen', 'caress', 'carve', 'cascade', 'cast', 'catch', 'cauterize', 'cave', 'center', 'chain', 'chant', 'char', 'charge', 'chart', 'chase', 'chisel', 'choke', 'choose', 'circle', 'clasp', 'cleave', 'climb', 'cloak', 'close', 'clot', 'clothe', 'cloud', 'clutch', 'coil', 'collapse', 'color', 'comb', 'combust', 'commune', 'compass', 'concede', 'condense', 'confess', 'conjure', 'console', 'consume', 'contain', 'contend', 'contour', 'converge', 'cool', 'copper', 'cork', 'corner', 'cradle', 'crack', 'crease', 'crest', 'crumble', 'crush', 'crust', 'crystallize', 'cup', 'curl', 'cure', 'cut', 'darken', 'dawn', 'dazzle', 'deafen', 'decay', 'declare', 'deepen', 'deglaze', 'deliver', 'delve', 'descend', 'desert', 'design', 'desire', 'desiccate', 'dissolve', 'distill', 'diverge', 'divide', 'doze', 'drag', 'draw', 'dredge', 'drift', 'drink', 'drip', 'drone', 'drown', 'dry', 'dull', 'dust', 'dwell', 'eclipse', 'edge', 'eddy', 'efface', 'elevate', 'embalm', 'embolden', 'embrace', 'embrittle', 'embroider', 'emerge', 'emote', 'empty', 'enamor', 'enclose', 'enkindle', 'enmesh', 'enshrine', 'ensnare', 'ensure', 'entwine', 'erode', 'erupt', 'escort', 'etch', 'evaporate', 'even', 'evidence', 'exhale', 'exile', 'expand', 'expose', 'exult', 'fade', 'fall', 'fan', 'fasten', 'fathom', 'feather', 'feed', 'fell', 'ferry', 'fester', 'fetch', 'fissure', 'fit', 'flake', 'flare', 'flash', 'flatten', 'flavor', 'flee', 'flicker', 'fling', 'flood', 'flourish', 'flow', 'flurry', 'flutter', 'fold', 'forage', 'forge', 'forget', 'form', 'fossilize', 'found', 'fracture', 'frame', 'freeze', 'freight', 'fret', 'fringe', 'fumble', 'fuse', 'gather', 'gaze', 'germinate', 'gild', 'gird', 'glimmer', 'glint', 'glisten', 'glitter', 'gloss', 'glove', 'glow', 'gnarl', 'gnaw', 'goad', 'gospel', 'graft', 'grain', 'grade', 'graze', 'grind', 'grip', 'grow', 'gulp', 'gutter', 'hail', 'half', 'halve', 'halo', 'hammer', 'hang', 'harden', 'harmonize', 'harvest', 'haunt', 'heave', 'hedge', 'heed', 'heft', 'hew', 'hinge', 'hiss', 'hitch', 'hoard', 'hold', 'hollow', 'hone', 'hook', 'hush', 'ignite', 'illumine', 'imbue', 'immerse', 'immure', 'impel', 'imprint', 'incise', 'include', 'increase', 'indent', 'infuse', 'inhale', 'inherit', 'ink', 'inlay', 'innervate', 'inoculate', 'inscribe', 'insinuate', 'inspire', 'interlace', 'interleave', 'interlock', 'intertwine', 'invoke', 'iron', 'irrigate', 'isolate', 'jar', 'join', 'kindle', 'knot', 'lacquer', 'lace', 'ladle', 'lag', 'lance', 'land', 'lap', 'lather', 'laugh', 'layer', 'lead', 'lean', 'leap', 'leaven', 'level', 'lick', 'lift', 'lighten', 'limn', 'linger', 'link', 'listen', 'live', 'load', 'loam', 'locate', 'lock', 'lodge', 'loom', 'loosen', 'lurch', 'lure', 'lull', 'lunge', 'mangle', 'mantle', 'mar', 'march', 'marry', 'mask', 'mass', 'match', 'meander', 'measure', 'meld', 'mellow', 'melt', 'mend', 'merge', 'mete', 'mill', 'mine', 'mirror', 'mist', 'mix', 'moan', 'molt', 'motion', 'mottle', 'mound', 'mourn', 'move', 'mull', 'mulch', 'murmur', 'nudge', 'nurse', 'obey', 'obscure', 'occasion', 'occlude', 'oil', 'open', 'orbit', 'ossify', 'outline', 'overarch', 'overcast', 'overflow', 'overgrow', 'overlay', 'overlap', 'overrun', 'overspill', 'pace', 'pack', 'paddle', 'pair', 'paint', 'pale', 'pall', 'pan', 'panel', 'paper', 'pare', 'part', 'pass', 'paste', 'pattern', 'pearl', 'peel', 'peer', 'pen', 'perforate', 'permeate', 'perch', 'percolate', 'perfect', 'perforate', 'perish', 'persist', 'personify', 'petrify', 'pierce', 'pile', 'pin', 'pinch', 'pine', 'pit', 'pivot', 'place', 'plaster', 'plate', 'pluck', 'plunge', 'poetize', 'point', 'polish', 'pool', 'pour', 'powder', 'praise', 'press', 'print', 'prise', 'prize', 'probe', 'prune', 'pull', 'pulse', 'punch', 'purge', 'purl', 'push', 'quiet', 'quiver', 'rake', 'ravel', 'raze', 'reassemble', 'reach', 'reap', 'reckon', 'redden', 'redeem', 'redraft', 'refashion', 'refine', 'refract', 'refresh', 'refracture', 'refrain', 'regard', 'reignite', 'rejoin', 'relight', 'relent', 'relieve', 'remain', 'remember', 'render', 'renew', 'reopen', 'repair', 'repeat', 'rephrase', 'replace', 'resound', 'resurface', 'restore', 'retract', 'retrieve', 'return', 'reveal', 'revise', 'ripple', 'rise', 'rivet', 'roil', 'roll', 'root', 'roughen', 'round', 'rouse', 'rub', 'ruminate', 'run', 'rush', 'rust', 'salt', 'sand', 'saturate', 'saw', 'scatter', 'scar', 'scavenge', 'scent', 'score', 'scour', 'scrawl', 'scrape', 'scratch', 'scream', 'screen', 'sculpt', 'sear', 'season', 'seat', 'secure', 'seed', 'seep', 'seethe', 'sequester', 'settle', 'sever', 'shade', 'shadow', 'shake', 'shape', 'shear', 'sheath', 'shelter', 'shimmer', 'shiver', 'shore', 'shoulder', 'shovel', 'shroud', 'sift', 'silence', 'silver', 'singe', 'sink', 'siphon', 'sketch', 'skim', 'skew', 'skid', 'slice', 'slide', 'sling', 'sluice', 'smelt', 'smolder', 'smudge', 'smooth', 'smother', 'snap', 'solder', 'soothe', 'sort', 'sound', 'span', 'spare', 'spark', 'spear', 'spell', 'spend', 'spike', 'spiral', 'splash', 'splice', 'split', 'spoil', 'sponge', 'sprout', 'spur', 'square', 'stack', 'stain', 'stake', 'stamp', 'stand', 'star', 'starch', 'startle', 'steep', 'stem', 'stitch', 'stoop', 'store', 'storm', 'straighten', 'strain', 'stray', 'streak', 'stress', 'stretch', 'strew', 'stride', 'strike', 'string', 'strip', 'stroke', 'strop', 'strum', 'submerge', 'suffuse', 'sugar', 'summon', 'suture', 'swallow', 'sway', 'swell', 'sweep', 'sweeten', 'swaddle', 'swathe', 'swirl', 'tack', 'tangle', 'taper', 'tattoo', 'temper', 'tender', 'tether', 'thaw', 'thicken', 'thread', 'thresh', 'thrill', 'thrive', 'throttle', 'throw', 'thrum', 'tick', 'tie', 'tilt', 'tincture', 'tinge', 'tinsel', 'tint', 'tip', 'tire', 'toast', 'toll', 'tong', 'torch', 'torque', 'trace', 'track', 'trade', 'trail', 'train', 'transfix', 'translate', 'transpose', 'trap', 'tread', 'tremble', 'trim', 'triple', 'truss', 'tuck', 'turn', 'twine', 'twist', 'unfurl', 'union', 'unlace', 'unmask', 'unravel', 'unveil', 'upend', 'uplift', 'uproot', 'urge', 'varnish', 'vault', 'veil', 'vein', 'vent', 'verge', 'verify', 'vest', 'vibrate', 'view', 'vine', 'visit', 'voice', 'wade', 'wane', 'warm', 'warp', 'wash', 'watermark', 'waste', 'watch', 'weave', 'wedge', 'weigh', 'weld', 'wet', 'whet', 'while', 'whisk', 'whisper', 'whiten', 'widen', 'wield', 'winnow', 'wire', 'wither', 'witness', 'wobble', 'wood', 'woo', 'wound', 'wreathe', 'wring', 'write', 'yoke'
            }
        return self._evocative_verbs

    @property
    def atmospheric_nouns(self) -> Set[str]:
        """Rich collection of atmospheric nouns for poetry"""
        if self._atmospheric_nouns is None:
            self._atmospheric_nouns = {
                'afterglow', 'alcove', 'alder', 'amber', 'antechamber', 'archway', 'ashes', 'aspen', 'aura', 'backwater', 'ballast', 'balcony', 'balm', 'basalt', 'bay', 'beacon', 'bedrock', 'bell', 'bevel', 'bivouac', 'blaze', 'bleak', 'bloom', 'bluff', 'bog', 'bole', 'borough', 'bough', 'boundary', 'braid', 'bramble', 'brass', 'breeze', 'brine', 'brim', 'brink', 'brume', 'brush', 'burl', 'burrow', 'butte', 'cairn', 'caldera', 'canopy', 'canyon', 'carapace', 'cask', 'casket', 'cellar', 'censer', 'chancel', 'channel', 'chaparral', 'char', 'charcoal', 'chassis', 'chimera', 'chisel', 'choir', 'chime', 'chiselmark', 'chord', 'chorus', 'cinder', 'cistern', 'cleft', 'clew', 'cliff', 'cloister', 'cloudbank', 'clove', 'clover', 'cobblestone', 'coda', 'coil', 'colonnade', 'compass', 'conifer', 'corbel', 'cornice', 'corridor', 'corral', 'cove', 'cradle', 'crag', 'cranberry', 'crawlspace', 'crease', 'crescent', 'crest', 'crib', 'crick', 'crinoline', 'crock', 'crossbeam', 'crown', 'crypt', 'crystal', 'cul-de-sac', 'culvert', 'cupola', 'current', 'curtain', 'cusp', 'cutbank', 'dale', 'damask', 'dapple', 'dark', 'deadfall', 'dell', 'delta', 'dew', 'diorama', 'diptych', 'ditch', 'dovecote', 'downpour', 'draught', 'drift', 'dune', 'dusk', 'ember', 'emptiness', 'enclave', 'escarpment', 'ether', 'eave', 'everglade', 'expanse', 'fable', 'facade', 'facet', 'fallow', 'fanlight', 'farmstead', 'fathom', 'fen', 'ferry', 'fieldstone', 'fissure', 'fjord', 'flame', 'flare', 'flange', 'flash', 'flax', 'flicker', 'floe', 'flora', 'flume', 'fog', 'foliage', 'foothill', 'forecourt', 'foredune', 'forest', 'forge', 'fork', 'fossil', 'fountain', 'foyer', 'fractal', 'frame', 'frost', 'fulcrum', 'fugue', 'furrow', 'gale', 'gallery', 'gantry', 'garden', 'gate', 'geyser', 'ghost', 'glade', 'glaze', 'gleam', 'glen', 'glimmer', 'gloom', 'gloss', 'glow', 'gneiss', 'gnomon', 'gorge', 'grotto', 'ground', 'grove', 'gulch', 'gulf', 'gypsum', 'hail', 'halo', 'hammock', 'harbor', 'harebell', 'haze', 'hearth', 'heather', 'hedge', 'heft', 'hemlock', 'hinge', 'hinterland', 'hive', 'holler', 'hollow', 'horizon', 'horn', 'hush', 'icefall', 'icicle', 'inlet', 'islet', 'ivy', 'jetty', 'kiln', 'kink', 'knell', 'knoll', 'lacuna', 'lagoon', 'lance', 'larder', 'lath', 'lattice', 'laurel', 'ledge', 'lee', 'legend', 'levee', 'lightwell', 'lilt', 'lintel', 'liqueur', 'lisp', 'loam', 'lobby', 'locus', 'loft', 'lozenge', 'lull', 'lumen', 'lumber', 'luster', 'lynchpin', 'lynx', 'mallow', 'marl', 'marsh', 'masonry', 'mast', 'maze', 'meadow', 'meander', 'mews', 'mica', 'midden', 'midthrash', 'mizzle', 'moor', 'moraine', 'morrow', 'moss', 'motherlode', 'mouth', 'muffle', 'mural', 'murk', 'muslin', 'myrrh', 'nacre', 'nave', 'nebula', 'nettle', 'nexus', 'nightfall', 'nimbus', 'notch', 'nunnery', 'oasis', 'oak', 'oath', 'obelisk', 'odyssey', 'outcrop', 'overhang', 'overstory', 'oxide', 'palisade', 'pall', 'pallet', 'palm', 'panhandle', 'panorama', 'parable', 'parapet', 'parch', 'parquet', 'pass', 'pasture', 'patch', 'path', 'pavilion', 'peat', 'pediment', 'pellicle', 'penumbra', 'perch', 'pergola', 'petal', 'pier', 'pieta', 'pierglass', 'pike', 'pillar', 'pinestraw', 'pinnacle', 'pith', 'planchet', 'plank', 'plateau', 'plume', 'pocket', 'pollen', 'pool', 'portal', 'portico', 'promontory', 'quarry', 'quaver', 'quay', 'quiet', 'quilt', 'quicksand', 'quoin', 'ravine', 'reed', 'refuge', 'reliquary', 'remnant', 'rift', 'rill', 'rim', 'rind', 'ringlet', 'ripple', 'riverbed', 'roadbed', 'robin', 'rockface', 'rood', 'roofline', 'room', 'rosette', 'rotunda', 'ruck', 'ruin', 'runnel', 'saddle', 'sage', 'sash', 'sconce', 'scree', 'script', 'scriptorium', 'scroll', 'scrub', 'seam', 'seamark', 'seascape', 'seat', 'sedge', 'seep', 'selvage', 'sepulcher', 'shadow', 'shaft', 'shale', 'shingle', 'shoal', 'shroud', 'sierra', 'sill', 'silt', 'sinew', 'siphon', 'skerry', 'sky', 'slack', 'slag', 'slake', 'slate', 'sleet', 'slope', 'sluice', 'smudge', 'snowmelt', 'solace', 'solstice', 'soot', 'sorrel', 'spandrel', 'spire', 'splinter', 'spring', 'spur', 'squall', 'stain', 'stairwell', 'stall', 'stand', 'starfield', 'steading', 'stem', 'steppe', 'stillness', 'stoa', 'stockade', 'stonework', 'stopgap', 'stove', 'strand', 'stratum', 'streamlet', 'streetlamp', 'striation', 'stubble', 'stud', 'sty', 'sumac', 'sump', 'sward', 'swell', 'swale', 'talus', 'tangle', 'tarn', 'tatter', 'teardrop', 'thicket', 'thimble', 'thistle', 'threshold', 'thrum', 'thundermug', 'tidal-flat', 'timber', 'tinsel', 'tint', 'tipple', 'tor', 'torchlight', 'torrent', 'torsion', 'towpath', 'tower', 'trace', 'tract', 'trailhead', 'trammel', 'transept', 'treeline', 'trellis', 'trench', 'truss', 'tundra', 'turpentine', 'twilight', 'umbra', 'undercroft', 'understory', 'upland', 'vale', 'valley', 'vanishing-point', 'vault', 'veldt', 'vellum', 'vespers', 'vestige', 'viga', 'vine', 'virga', 'vista', 'void', 'wadi', 'wake', 'wall', 'ward', 'wash', 'wasteland', 'waterline', 'watershed', 'wave', 'way', 'weald', 'weir', 'weld', 'well', 'weirwood', 'westerly', 'wetland', 'wheatfield', 'whorl', 'wildflower', 'willow', 'windbreak', 'window', 'windrow', 'windshift', 'wintering', 'wisp', 'wold', 'woodsmoke', 'workbench', 'wreath', 'wreck', 'wren', 'wrack', 'yard', 'yearling', 'yew', 'yonder', 'zephyr'
            }
        return self._atmospheric_nouns

    @property
    def emotional_keywords(self) -> Dict[str, Set[str]]:
        """Emotional tone keywords organized by category"""
        if self._emotional_keywords is None:
            self._emotional_keywords = {
                'dark': {'death', 'dark', 'shadow', 'fear', 'pain', 'sorrow', 'lost', 'empty', 'cold', 'despair', 'grief', 'melancholy', 'lonely', 'desolate', 'bleak', 'grim', 'haunted', 'broken', 'shattered', 'void'},
                'light': {'bright', 'warm', 'hope', 'joy', 'love', 'gentle', 'soft', 'golden', 'radiant', 'gleaming', 'luminous', 'serene', 'peaceful', 'blessed', 'divine', 'pure', 'innocent', 'tender', 'sweet', 'blissful'},
                'dynamic': {'crash', 'slam', 'rush', 'strike', 'burst', 'shatter', 'thunder', 'storm', 'rage', 'fierce', 'wild', 'turbulent', 'violent', 'explosive', 'dramatic', 'intense', 'powerful', 'forceful', 'electric', 'blazing'},
                'quiet': {'whisper', 'silence', 'still', 'calm', 'peace', 'rest', 'fade', 'hush', 'murmur', 'gentle', 'soft', 'mellow', 'tranquil', 'serene', 'placid', 'subdued', 'muted', 'delicate', 'subtle', 'ethereal'},
                'mysterious': {'unknown', 'hidden', 'secret', 'strange', 'mysterious', 'unseen', 'enigmatic', 'cryptic', 'veiled', 'shadowy', 'elusive', 'obscure', 'mystical', 'arcane', 'puzzling', 'inscrutable', 'unfathomable', 'otherworldly', 'spectral', 'phantom'}
            }
        return self._emotional_keywords

    @property
    def poetic_attributes(self) -> Dict[str, Set[str]]:
        """Organized poetic attributes by category"""
        if self._poetic_attributes is None:
            self._poetic_attributes = {
                'physical': {'rough', 'smooth', 'sharp', 'soft', 'cold', 'warm', 'dense', 'light', 'heavy', 'delicate', 'solid', 'liquid', 'crystalline', 'metallic', 'wooden', 'silken'},
                'temporal': {'fleeting', 'eternal', 'sudden', 'gradual', 'ancient', 'fresh', 'recurring', 'momentary', 'endless', 'brief', 'timeless', 'ephemeral', 'lasting', 'transient', 'permanent', 'cyclical'},
                'emotional': {'bitter', 'sweet', 'painful', 'gentle', 'fierce', 'tender', 'raw', 'passionate', 'melancholic', 'ecstatic', 'serene', 'turbulent', 'longing', 'yearning', 'content', 'restless'},
                'visual': {'bright', 'dim', 'clear', 'hazy', 'vivid', 'pale', 'shadowy', 'gleaming', 'luminous', 'radiant', 'glowing', 'sparkling', 'shimmering', 'translucent', 'opaque', 'iridescent'},
                'abstract': {'mysterious', 'complex', 'simple', 'profound', 'shallow', 'vast', 'intimate', 'infinite', 'bounded', 'expansive', 'concentrated', 'diffuse', 'essential', 'fundamental', 'superficial', 'deep'},
                'motion': {'restless', 'still', 'flowing', 'rigid', 'dancing', 'heavy', 'weightless', 'spiraling', 'cascading', 'drifting', 'rushing', 'meandering', 'soaring', 'sinking', 'floating', 'tumbling'}
            }
        return self._poetic_attributes

    def is_evocative_word(self, word: str) -> bool:
        """Check if a word is evocative (verb or atmospheric noun)"""
        word_lower = word.lower()
        return word_lower in self.evocative_verbs or word_lower in self.atmospheric_nouns

    def get_emotional_tone(self, text: str) -> str:
        """Detect emotional tone of text"""
        text_lower = text.lower()
        for tone, keywords in self.emotional_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return tone
        return 'neutral'

    def get_random_attributes(self, category: str = None, count: int = 2) -> List[str]:
        """Get random poetic attributes from a category"""
        import random

        if category and category in self.poetic_attributes:
            return random.sample(list(self.poetic_attributes[category]), min(count, len(self.poetic_attributes[category])))
        else:
            # Mix from all categories
            all_attributes = []
            for attrs in self.poetic_attributes.values():
                all_attributes.extend(attrs)
            return random.sample(all_attributes, min(count, len(all_attributes)))


# Global instance for easy access
vocabulary = SharedVocabulary()