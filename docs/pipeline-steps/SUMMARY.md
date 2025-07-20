# Pipeline Documentation Summary

We've created a comprehensive documentation structure for the Black Marble pipeline with the following components:

## Structure

```
docs/pipeline-steps/
├── README.md                    # Overview and navigation
├── viirs-preprocessing.md       # VIIRS crop, resample, reproject
├── landsat-qa-masking.md       # Cloud/shadow detection strategies  
├── spectral-indices.md         # NDVI/NDWI with epsilon handling
├── ndui-calculation.md         # Urban index with water masking
└── urban-field-enhancement.md  # Multi-scale road influence fields
```

## Key Features

### 1. Clear Navigation
- Main README links to detailed docs for processing steps
- I/O steps noted but not expanded (no decisions needed)
- Visual indicators (⚙️) for processing steps with decisions

### 2. Consistent Documentation Pattern
Each processing step document includes:
- **Overview**: What and why
- **Algorithm**: Technical details with code
- **Key Decisions**: Parameters and trade-offs
- **Test Coverage**: Links to specific test files and methods
- **Common Issues**: Troubleshooting guide
- **Configuration**: How to customize behavior
- **Related Steps**: Pipeline context

### 3. Test Traceability
Every processing function is linked to its tests:
```
Pipeline Function → Documentation → Test File
crop_to_bounds() → viirs-preprocessing.md → test_pipeline_flow.py::test_03_viirs_processing
```

### 4. Decision Documentation
Critical decisions are explained with rationale:
- VIIRS resampling to 12m (not 30m) for urban detail
- QA masking strategies (conservative/moderate/permissive)
- Epsilon handling with `np.copysign()` for indices
- Multi-scale parameters for urban fields

## Benefits

1. **Onboarding**: New developers can understand the pipeline flow and decisions
2. **Debugging**: Each step has common issues and solutions
3. **Testing**: Clear mapping from functionality to test coverage
4. **Customization**: Configuration options documented with trade-offs

## Next Steps

To complete the documentation:
1. Add remaining processing steps (temporal compositing, contrast enhancement, etc.)
2. Create similar docs for auxiliary features (statistics, validation)
3. Add example outputs/visualizations
4. Link from main project README

This structure makes the pipeline behavior transparent and testable while keeping documentation close to the code it describes.