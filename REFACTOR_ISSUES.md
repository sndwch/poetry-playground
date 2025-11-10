# Core Refactor Issue Log

## Phase 1.3: Internal Core Module Import Issues

### Issue 1: Core modules importing from parent (non-core) modules
**Status:** ✅ RESOLVED (Phase 1.3)

Core modules have relative imports (`.`) that need to become parent imports (`..`) for modules outside core/:

| File | Line | Current Import | Needs To Change To |
|------|------|----------------|-------------------|
| `lexigen.py` | 6 | `from .cache import` | `from ..cache import` |
| `lexigen.py` | 7 | `from .datamuse_api import` | `from ..datamuse_api import` |
| `lexigen.py` | 8 | `from .utils import` | `from ..utils import` |
| `quality_scorer.py` | 30 | `from .logger import logger` | `from ..logger import logger` |
| `pos_vocabulary.py` | 20 | `from .forms import count_syllables` | `from ..forms import count_syllables` |
| `pos_vocabulary.py` | 21 | `from .logger import logger` | `from ..logger import logger` |
| `document_library.py` | 17 | `from .config import` | `from ..config import` |
| `word_validator.py` | 24 | `from .logger import logger` | `from ..logger import logger` |

**Note:** Imports of core modules from within core (e.g., `from .quality_scorer`) remain unchanged - they're siblings.

### Issue 2: pos_vocabulary.py imports from vocabulary
**Status:** VERIFIED OK

Line 22: `from .vocabulary import vocabulary` - Both files are in core/, so this is fine.

---

## Phase 1.6: Data Path Issues

### Issue 3: QualityScorer data directory path broken after move to core/
**Status:** ✅ RESOLVED (Phase 1.6)

**Location:** `poetryplayground/core/quality_scorer.py:142`

**Problem:**
- **Before move:** `quality_scorer.py` was in `poetryplayground/`
  - `Path(__file__).parent / "data"` → `poetryplayground/data/` ✓
- **After move:** `quality_scorer.py` is in `poetryplayground/core/`
  - `Path(__file__).parent / "data"` → `poetryplayground/core/data/` ✗ (doesn't exist!)

**Data files affected:**
- `poetry_cliches.json` (6,148 bytes)
- `concreteness_ratings.txt` (3,278 bytes)

**Test failures caused:**
- `test_quality_scorer.py::test_initialization` - expects concreteness_cache loaded
- `test_quality_scorer.py::test_concreteness_scoring` - needs concreteness data
- `test_quality_scorer.py::test_imagery_scoring_with_context` - needs concreteness data
- `test_quality_scorer.py::test_database_loading` - expects >100 cliche phrases, gets 6 (fallback)

**Fix:**
Change line 142 from:
```python
data_dir = Path(__file__).parent / "data"
```
to:
```python
data_dir = Path(__file__).parent.parent / "data"
```

This navigates: `core/quality_scorer.py` → `core/` → `poetryplayground/` → `poetryplayground/data/` ✓

### Issue 4: Lexicon snapshot path broken after move to core/
**Status:** ✅ RESOLVED (Phase 1.6)

**Location:** `poetryplayground/core/lexicon.py:150-152`

**Problem:**
- **Before move:** `lexicon.py` was in `poetryplayground/`
  - `module_dir.parent` → project root ✓
  - `tests/data/snapshot_lexicon.txt` found ✓
- **After move:** `lexicon.py` is in `poetryplayground/core/`
  - `module_dir.parent` → `poetryplayground/` (not project root!) ✗
  - Looks for `poetryplayground/tests/` which doesn't exist ✗

**Fix:**
Need one more `.parent` call:
```python
module_dir = Path(__file__).parent          # poetryplayground/core/
package_dir = module_dir.parent             # poetryplayground/
project_root = package_dir.parent           # (actual project root)
snapshot_path = project_root / "tests" / "data" / "snapshot_lexicon.txt"
```

---

## Fixing Strategy
1. Update each file's imports from parent modules ✅
2. Keep sibling imports (within core/) as-is ✅
3. Fix data directory paths in moved files 🔄
4. Test after all fixes complete 🔄
