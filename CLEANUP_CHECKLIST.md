# VEDA Black Marble Cleanup Checklist

## Overview
This checklist tracks the systematic cleanup of ~7,000 lines of dead code and architectural issues identified in PR #8 review.

## Phase 1: Baseline & Tooling ✅
**Goal:** Establish metrics and tooling infrastructure
**Timeline:** Week 1

### Setup Tasks
- [x] Create feature branch `cleanup/phase-1-baseline`
- [x] Add development dependencies to `pyproject.toml`:
  - [x] vulture
  - [x] autoflake
  - [x] interrogate
  - [x] radon
- [x] Configure tools in `pyproject.toml`:
  - [x] vulture settings
  - [x] autoflake settings
  - [x] pyright strict mode (replaced mypy)
- [x] Create `scripts/` directory for audit scripts

### Baseline Metrics
- [x] Run vulture baseline: Created `.metrics/vulture_2025-07-18.json`
- [x] Generated baseline metrics: 9 vulture items, 201 ruff issues
- [ ] Generate current coverage report: `pytest --cov=blackmarble --cov-report=html`
- [ ] Count current LOC: Create `scripts/audit_metrics.py`
- [ ] Generate import graph with pydeps
- [ ] Document current test count and pass rate

### Scripts to Create
- [ ] `scripts/audit_metrics.py` - LOC counter and tracker
- [x] `scripts/find_pointless_wrappers.py` - Wrapper detector
- [x] `scripts/audit_dead_code.py` - JSON inventory generator (vulture)
- [x] `scripts/audit_pyright.py` - Type checking audit
- [x] `scripts/audit_ruff.py` - Linting audit
- [ ] `scripts/vulture_whitelist.py` - False positive allowlist

### Pre-commit Setup
- [x] Add `.pre-commit-config.yaml` with:
  - [x] ruff (warning mode)
  - [x] pyright (replaced mypy)
  - [x] vulture (via autoflake, warning mode)
  - [x] black (manual stage)
  - [x] isort (manual stage)
- [ ] Run `pre-commit install`

### CI/CD Setup
- [x] Create `.github/workflows/cleanup-gates.yml`:
  - [x] Non-blocking diagnostics job
  - [x] Runs all audit scripts
  - [x] Uploads metrics as artifacts
  - [x] Uses uv for dependency management

**Commit:** "Add cleanup tooling and baseline metrics" ✅

**Results:**
- Baseline: 9 dead code items (vulture), 201 linting issues (ruff)
- Tools configured: black, isort, ruff, vulture, autoflake, pyright
- All tools in warning/non-blocking mode

---

## Phase 2: Dead Code Purge 🗑️
**Goal:** Remove obvious dead files and unused functions
**Timeline:** Week 2

### Dead Code Removal ✅
- [x] Remove `if False` blocks (removed 147 lines)
- [x] Remove unused variables (fixed all F841 errors in main code)
- [x] Remove unused imports (fixed async_orchestrator, async_base)
- [x] Comment out unused enhancement code sections
- [x] Reduce total issues from 201 to 37 (81.6% reduction)

### Whole File Deletions
- [ ] Delete backup/old files:
  - [ ] `blackmarble/acquire/viirs_bloated.py.backup`
  - [ ] `blackmarble/acquire/async_viirs.py.old`
  - [ ] `blackmarble/acquire/async_viirs_functional.py`
- [ ] Delete unused modules:
  - [ ] `blackmarble/prepare/quality.py` (499 lines)
- [ ] Search and remove any `*.old`, `*.backup`, `*.deprecated` files

### Verification for Each Deletion
- [ ] Run `rg "filename" .` to check no imports remain
- [ ] Run `rg "from module import" .` for module imports
- [ ] Check `__init__.py` exports
- [ ] Run test suite after each deletion

### Orphan Function Removal
- [x] Run `vulture blackmarble > vulture_report.txt` (9 items found)
- [ ] For each reported unused function:
  - [ ] Verify with ripgrep it's truly unused
  - [ ] Check if it's in `__all__` exports
  - [ ] Delete if confirmed unused
  - [ ] Or add to deprecation list if public API

### Broken Code Fixes
- [x] Run `pyright blackmarble` (replaced mypy)
- [x] Fix undefined variable errors
- [ ] Remove functions with syntax errors

### Update Tests
- [ ] Remove tests for deleted code
- [ ] Ensure coverage doesn't drop below baseline

**Commits:** 
- "Remove if False blocks and unused VIIRS quality masking" ✅
- "Remove unused variables and clean up dead code" ✅

---

## Phase 3: Remove Pointless Wrappers 🎁
**Goal:** Eliminate zero-value wrapper functions
**Timeline:** Week 3

### Wrapper Detection
- [ ] Run `tools/find_pointless_wrappers.py`
- [ ] Create inventory in `docs/cleanup_tracker.md`
- [ ] Classify each wrapper:
  - [ ] DELETE - No added value
  - [ ] KEEP - Adds logging/retry/validation
  - [ ] INLINE - Replace at call sites

### Common Wrapper Patterns to Check
- [ ] `load_with_rasterio()` → Use `rasterio.open()` directly
- [ ] `save_with_rasterio()` → Use `rasterio.open(mode='w')` directly
- [ ] `create_virtual_raster()` → Use `gdal.BuildVRT()` directly
- [ ] Path manipulation wrappers → Use pathlib directly
- [ ] Single-line numpy wrappers → Use numpy directly

### Module-by-Module Review
- [ ] `blackmarble/export/cog.py` - Check for rasterio wrappers
- [ ] `blackmarble/enhance/contrast.py` - Verify adds real logic
- [ ] `blackmarble/prepare/spatial.py` - Remove redundant helpers
- [ ] Any `utils` modules - Common location for wrappers

### Inlining Process
- [ ] For each wrapper to remove:
  - [ ] Find all call sites with ripgrep
  - [ ] Replace with direct library call
  - [ ] Update imports
  - [ ] Run tests
  - [ ] Delete wrapper function

**Commit:** "Remove pointless wrapper functions"

---

## Phase 4: De-duplicate & Consolidate 🔄
**Goal:** Single implementation per feature
**Timeline:** Week 4

### VIIRS Consolidation
- [ ] Decide on canonical implementation: `acquire/viirs.py`
- [ ] Extract useful async logic to internal helper
- [ ] Create `acquire/viirs_types.py` for shared types
- [ ] Update `pipeline_async.py` to use consolidated version
- [ ] Delete redundant implementations:
  - [ ] `async_viirs_functional.py`
  - [ ] Other async variants

### Spatial Operations
- [ ] Audit all spatial helpers across modules
- [ ] Keep only in `prepare/spatial.py`:
  - [ ] `reproject_image`
  - [ ] `resample_image` 
  - [ ] `crop_to_bounds`
  - [ ] `align_rasters`
- [ ] Remove duplicates from other modules
- [ ] Ensure consistent parameter names/types

### Component Pattern Review
- [ ] Count concrete implementations per abstract class
- [ ] Remove abstractions with 0-1 implementations
- [ ] Simplify over-engineered patterns

### Public API Definition
- [ ] Update all `__init__.py` with explicit `__all__`
- [ ] Add `from __future__ import annotations`
- [ ] Document public vs private modules
- [ ] Create deprecation utilities if needed

**Commit:** "Consolidate duplicate implementations"

---

## Phase 5: Automated Verification ✅
**Goal:** Prevent regression with CI gates
**Timeline:** Week 5

### CI Gates - Make Failing
- [ ] Update `.github/workflows/cleanup-gates.yml`:
  - [ ] vulture - must exit 0
  - [ ] mypy --strict - must pass
  - [ ] coverage - set threshold at current + 5%
  - [ ] ruff - zero errors

### Pre-commit - Make Blocking  
- [ ] Update `.pre-commit-config.yaml`:
  - [ ] vulture - fail on dead code
  - [ ] mypy - fail on type errors
  - [ ] ruff - fail on style issues

### Monitoring Setup
- [ ] Add metrics tracking to CI:
  - [ ] LOC count per module
  - [ ] Coverage percentage
  - [ ] Number of vulture warnings
  - [ ] Type coverage percentage

### Documentation
- [ ] Create `CONTRIBUTING.md` section on:
  - [ ] When wrappers are acceptable
  - [ ] Dead code prevention
  - [ ] Architecture decisions
- [ ] Update README with new requirements

**Commit:** "Add strict CI gates for code quality"

---

## Phase 6: Documentation & Release 📚
**Goal:** Document changes and prepare release
**Timeline:** Week 6

### Deprecation Notices
- [ ] Create `blackmarble/utils/deprecation.py`
- [ ] Mark any public APIs for deprecation
- [ ] Add deprecation warnings with version info

### Documentation Updates
- [ ] Update `CHANGELOG.md` with:
  - [ ] Removed modules list
  - [ ] Breaking changes
  - [ ] Migration guide
- [ ] Create `docs/architecture/decisions/001-cleanup.md`
- [ ] Update API documentation

### Release Preparation
- [ ] Tag current version before cleanup
- [ ] Run full test suite
- [ ] Test all example notebooks
- [ ] Create release candidate
- [ ] Monitor for 7 days before release

**Commit:** "Document cleanup and prepare release"

---

## Success Metrics 📊
- [ ] ~7,000 LOC removed
- [ ] Zero vulture warnings (excluding whitelist)
- [ ] 90%+ test coverage
- [ ] All mypy --strict checks pass
- [ ] Build time reduced by 25%
- [ ] Single implementation per feature

## Tracking
- Total LOC at start: _____
- Total LOC at end: _____  
- Test coverage start: _____%
- Test coverage end: _____%
- Vulture warnings start: _____
- Vulture warnings end: 0