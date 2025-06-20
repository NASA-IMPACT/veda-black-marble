#!/usr/bin/env python3
"""Minimal test to figure out correct osmnx API usage."""

import osmnx as ox

print(f"osmnx version: {ox.__version__}")

# Test 1: Try the old API style (north, south, east, west)
print("\nTest 1: Old API style (north, south, east, west)")
try:
    G = ox.graph_from_bbox(40.7580, 40.7570, -73.9850, -73.9860)
    print(f"✓ Success! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 2: Try with bbox tuple
print("\nTest 2: bbox tuple (north, south, east, west)")
try:
    G = ox.graph_from_bbox(bbox=(40.7580, 40.7570, -73.9850, -73.9860))
    print(f"✓ Success! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 3: Try different parameter order
print("\nTest 3: bbox tuple (west, south, east, north)")
try:
    G = ox.graph_from_bbox(bbox=(-73.9860, 40.7570, -73.9850, 40.7580))
    print(f"✓ Success! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

# Test 4: Check what parameters the function expects
print("\nTest 4: Checking function signature")
import inspect
sig = inspect.signature(ox.graph_from_bbox)
print(f"Function signature: {sig}")

# Test 5: Use the graph_from_place as alternative
print("\nTest 5: Alternative - graph_from_point")
try:
    G = ox.graph_from_point((40.7575, -73.9855), dist=100)  # 100m radius
    print(f"✓ Success! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")