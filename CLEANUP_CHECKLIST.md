# VEDA Black Marble Cleanup Checklist

## Overview
This checklist tracks the systematic cleanup of ~7,000 lines of dead code and architectural issues identified in PR #8 review.

## Phase 1: Baseline & Tooling ⏳
**Goal:** Establish metrics and tooling infrastructure
**Timeline:** Week 1

### Setup Tasks
- [ ] Create feature branch `cleanup/phase-1-baseline`
- [ ] Add development dependencies to `pyproject.toml`:
  - [ ] vulture
  - [ ] autoflake
  - [ ] interrogate
  - [ ] radon
- [ ] Configure tools in `pyproject.toml`:
  - [ ] vulture settings
  - [ ] autoflake settings
  - [ ] mypy strict mode
- [ ] Create `tools/` directory for audit scripts

### Baseline Metrics
- [ ] Generate current coverage report: `pytest --cov=blackmarble --cov-report=html`
- [ ] Save baseline to `docs/metrics/baseline_coverage_YYYYMMDD.html`
- [ ] Run vulture baseline: `vulture blackmarble > docs/metrics/vulture_baseline.txt`
- [ ] Count current LOC: Create `tools/audit_metrics.py`
- [ ] Generate import graph with pydeps
- [ ] Document current test count and pass rate

### Scripts to Create
- [ ] `tools/audit_metrics.py` - LOC counter and tracker
- [ ] `tools/find_pointless_wrappers.py` - Wrapper detector
- [ ] `tools/dead_code_audit.py` - JSON inventory generator
- [ ] `tools/vulture_whitelist.py` - False positive allowlist

### Pre-commit Setup
- [ ] Add `.pre-commit-config.yaml` with:
  - [ ] ruff
  - [ ] mypy
  - [ ] vulture (warning only initially)
- [ ] Run `pre-commit install`

### CI/CD Setup
- [ ] Create `.github/workflows/cleanup-gates.yml`:
  - [ ] lint job (ruff)
  - [ ] type-check job (mypy)
  - [ ] dead-code job (vulture - warning only)
  - [ ] coverage job (pytest with threshold)

**Commit:** "Add cleanup tooling and baseline metrics"

---

## Phase 2: Dead Code Purge 🗑️
**Goal:** Remove obvious dead files and unused functions
**Timeline:** Week 2

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
- [ ] Run `vulture blackmarble > vulture_report.txt`
- [ ] For each reported unused function:
  - [ ] Verify with ripgrep it's truly unused
  - [ ] Check if it's in `__all__` exports
  - [ ] Delete if confirmed unused
  - [ ] Or add to deprecation list if public API

### Broken Code Fixes
- [ ] Run `mypy --strict blackmarble` 
- [ ] Fix undefined variable errors
- [ ] Remove functions with syntax errors

### Update Tests
- [ ] Remove tests for deleted code
- [ ] Ensure coverage doesn't drop below baseline

**Commit:** "Remove dead code and unused modules"

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