#!/usr/bin/env python3
"""Test the mask logic to understand why all pixels are being masked out."""

import numpy as np
from blackmarble.masks import check_values, mark_cloud, mark_aerosol, mark_radsat

# Test check_values function
print("Testing check_values function:")
test_array = np.array([1, 2, 3, 4, 5])
values_to_mask = [2, 4]
result = check_values(values_to_mask, test_array)
print(f"Input: {test_array}")
print(f"Values to mask: {values_to_mask}")
print(f"Result: {result}")
print(f"Expected: [1. 0. 1. 0. 1.] (1=keep, 0=mask)")
print()

# Test cloud mask with common QA_PIXEL values
print("Testing mark_cloud function:")
# Common QA_PIXEL values:
# 21824 = Clear land
# 21952 = Clear water
# 54596 = Cloud
# 55052 = Cloud shadow
test_qa_pixel = np.array([21824, 21952, 54596, 55052, 22280])
cloud_mask = mark_cloud(test_qa_pixel)
print(f"QA_PIXEL values: {test_qa_pixel}")
print(f"Cloud mask result: {cloud_mask}")
print(f"Expected: [1. 1. 0. 0. 0.] (clear pixels kept)")
print()

# Test aerosol mask
print("Testing mark_aerosol function:")
test_qa_aerosol = np.array([96, 192, 194, 196, 228, 0])
aerosol_mask = mark_aerosol(test_qa_aerosol)
print(f"QA_AEROSOL values: {test_qa_aerosol}")
print(f"Aerosol mask result: {aerosol_mask}")
print(f"Expected: [1. 0. 0. 0. 0. 1.] (good aerosol kept)")
print()

# Test radsat mask
print("Testing mark_radsat function:")
# QA_RADSAT uses bit flags:
# Bit 2 (value 4) = Band 3 saturated
# Bit 3 (value 8) = Band 4 saturated
# Bit 4 (value 16) = Band 5 saturated
test_qa_radsat = np.array([0, 4, 8, 12, 16, 28])  # 12=4+8, 28=4+8+16
b3_mask, b4_mask, b5_mask = mark_radsat(test_qa_radsat)
print(f"QA_RADSAT values: {test_qa_radsat}")
print(f"Band 3 radsat mask: {b3_mask}")
print(f"Band 4 radsat mask: {b4_mask}")
print(f"Band 5 radsat mask: {b5_mask}")
print("Expected B3: [1. 0. 1. 0. 1. 0.] (saturated pixels masked)")
print("Expected B4: [1. 1. 0. 0. 1. 0.]")
print("Expected B5: [1. 1. 1. 1. 0. 0.]")
print()

# Test the actual logic error in radsat
print("Debugging radsat bitwise operations:")
test_val = 12  # Binary: 1100 (bits 2 and 3 set)
print(f"Value {test_val} in binary: {bin(test_val)}")
print(f"test_val & 4 = {test_val & 4} (should be 4 if bit 2 is set)")
print(f"test_val & 8 = {test_val & 8} (should be 8 if bit 3 is set)")
print(f"test_val & 16 = {test_val & 16} (should be 16 if bit 4 is set)")

# The issue might be in how check_values is called
print("\nDEBUG: Testing the actual mark_radsat logic:")
r_px = np.array([0, 4, 8, 12])
print(f"r_px: {r_px}")
print(f"r_px & 4: {r_px & 4}")
print(f"check_values([4], r_px & 4): {check_values([4], r_px & 4)}")