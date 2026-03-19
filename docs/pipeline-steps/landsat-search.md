# Landsat Scene Search and Selection

## Overview

The pipeline finds the best Landsat scenes for temporal consistency by searching within time windows and grouping scenes by acquisition date rather than selecting tiles individually.

## Key Strategy

**Date-Grouped Search**: Instead of selecting the "best" scene per tile (which mixes dates), the system:
1. Searches within a time window (default ±7 days) around target dates
2. Groups scenes by acquisition date with tolerance (12 days)
3. Selects complete date groups with lowest cloud cover
4. Prioritizes temporal consistency over individual tile quality

## Key Parameters

**Function**: `search_best_scenes()` in `landsat_search_grouped.py`

- **window_days** (7): Search ±7 days from each target date
- **tolerance_days** (12.0): Group scenes within 12 days together
- **allow_incomplete** (False): Require 100% spatial coverage

## Configuration

- Defaults: `window_days=7`, `tolerance_days=12.0`.
- Larger areas or sparse coverage can use broader windows and tolerance.

## Decision Trade-offs

- **Smaller window_days**: Better temporal consistency, fewer cloud-free options
- **Larger tolerance_days**: Handles multi-day acquisitions, less temporal precision
- **allow_incomplete=True**: Returns partial coverage instead of failing

## Pipeline Integration

The pipeline automatically creates 12 target dates at ~30-day intervals and applies grouped search to each date. This ensures temporal composites use scenes from consistent time periods rather than mixing dates within individual composites.

## Related Steps

- Previous: Data acquisition setup
- Next: [QA Masking](./landsat-qa-masking.md) - Apply cloud masks
- Outputs to: All Landsat processing steps