# Extending Black Marble Components

This guide shows how to create custom components and extend the Black Marble pipeline.

## Creating a New Component

All components should implement the base component interface:

```python
from blackmarble.components.base import ComposableComponent, MaskResult
import numpy as np

class AtmosphericCorrector(ComposableComponent):
    """Apply atmospheric correction to imagery."""
    
    def __init__(
        self,
        name: str = "atmospheric_corrector",
        correction_method: str = "dark_object",
        **kwargs
    ):
        super().__init__(name)
        self.correction_method = correction_method
        self._config = {
            "correction_method": correction_method,
            **kwargs
        }
    
    def process(self, data: np.ndarray, **kwargs) -> MaskResult:
        """Apply atmospheric correction.
        
        Args:
            data: Input imagery data
            **kwargs: Additional parameters
            
        Returns:
            MaskResult with corrected data in metadata
        """
        # Implement correction algorithm
        corrected = self._apply_correction(data, **kwargs)
        
        # Create quality mask (all pixels valid after correction)
        mask = np.ones_like(data, dtype=float)
        
        # Generate statistics
        stats = {
            "mean_correction": float(np.mean(corrected - data)),
            "max_correction": float(np.max(np.abs(corrected - data))),
        }
        
        return MaskResult(
            mask=mask,
            statistics=stats,
            metadata={
                "corrected_data": corrected,
                "correction_method": self.correction_method,
            }
        )
    
    def _apply_correction(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the actual correction algorithm."""
        if self.correction_method == "dark_object":
            # Simple dark object subtraction
            dark_value = np.percentile(data[data > 0], 1)
            return np.maximum(data - dark_value, 0)
        else:
            raise ValueError(f"Unknown correction method: {self.correction_method}")
```

## Adding Components to the Pipeline

### 1. Add to Pipeline Context

```python
# In pipeline_refactored.py or your custom pipeline
@dataclass
class PipelineContext:
    # ... existing fields ...
    atmospheric_corrector: AtmosphericCorrector
    
    @classmethod
    def create(cls, **kwargs):
        # ... existing initialization ...
        atmospheric_corrector = AtmosphericCorrector()
        
        return cls(
            # ... existing components ...
            atmospheric_corrector=atmospheric_corrector,
        )
```

### 2. Use in Pipeline Phase

```python
def _atmospheric_correction_phase(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply atmospheric correction to all bands."""
    corrected_data = {}
    
    for band_name, band_data in data.items():
        result = self.ctx.atmospheric_corrector.process(
            band_data,
            band_name=band_name,
            sensor="landsat"
        )
        
        # Extract corrected data from metadata
        if result.metadata and "corrected_data" in result.metadata:
            corrected_data[band_name] = result.metadata["corrected_data"]
        else:
            corrected_data[band_name] = band_data
            
    return corrected_data
```

## Component Best Practices

### 1. Configuration

- Accept all configuration through constructor parameters
- Store configuration in `self._config` dictionary
- Provide sensible defaults

### 2. Error Handling

```python
def process(self, data: np.ndarray, **kwargs) -> MaskResult:
    # Validate inputs
    if data.ndim not in [2, 3]:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
    
    try:
        result = self._process_implementation(data, **kwargs)
    except Exception as e:
        # Log error and return empty mask
        logger.error(f"Processing failed: {e}")
        return MaskResult(
            mask=np.zeros_like(data),
            statistics={"error": str(e)},
            metadata=None
        )
```

### 3. Statistics and Metadata

Always provide useful statistics and metadata:

```python
return MaskResult(
    mask=output_mask,
    statistics={
        "valid_pixels": int(np.sum(output_mask > 0)),
        "total_pixels": int(output_mask.size),
        "valid_percentage": float(100.0 * np.sum(output_mask > 0) / output_mask.size),
    },
    metadata={
        "processing_time": elapsed_time,
        "parameters": self._config,
        "output_data": processed_data,  # If needed by downstream components
    }
)
```

### 4. Type Hints

Use modern Python type hints:

```python
def process(
    self, 
    data: np.ndarray,
    quality_mask: np.ndarray | None = None,
    **kwargs: Any
) -> MaskResult:
    """Process with optional quality mask."""
```

## Testing Custom Components

Create comprehensive tests for your components:

```python
import pytest
import numpy as np
from your_module import AtmosphericCorrector

class TestAtmosphericCorrector:
    def test_dark_object_correction(self):
        # Create test data with known properties
        data = np.array([[100, 110, 120], [105, 115, 125], [102, 112, 122]])
        
        corrector = AtmosphericCorrector(correction_method="dark_object")
        result = corrector.process(data)
        
        # Check results
        assert result.metadata is not None
        corrected = result.metadata["corrected_data"]
        
        # Dark object should be subtracted
        assert np.all(corrected < data)
        assert np.all(corrected >= 0)
    
    def test_invalid_method(self):
        corrector = AtmosphericCorrector(correction_method="invalid")
        with pytest.raises(ValueError, match="Unknown correction method"):
            corrector.process(np.ones((10, 10)))
```

## Performance Considerations

### Memory Efficiency

```python
def process_large_data(self, data: np.ndarray, chunk_size: int = 1000) -> MaskResult:
    """Process large arrays in chunks to save memory."""
    if data.shape[0] > chunk_size:
        # Process in chunks
        results = []
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i+chunk_size]
            chunk_result = self._process_chunk(chunk)
            results.append(chunk_result)
        
        # Combine results
        return self._combine_results(results)
    else:
        return self._process_chunk(data)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

def process_parallel(self, data_list: list[np.ndarray]) -> list[MaskResult]:
    """Process multiple datasets in parallel."""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(self.process, data_list))
    return results
```