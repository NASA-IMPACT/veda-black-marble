#!/usr/bin/env python3
"""
CLI wrapper for the Black Marble pipeline.
Run this from the project root to execute the full workflow with environment variables.
"""
import sys

from blackmarble.pipeline import main

if __name__ == '__main__':
    sys.exit(main())