#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Diabetic Retinopathy Detection
File: __init__.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-10-02
Updated: 2025-10-02
License: MIT License (see LICENSE file for details)
=====================================================================================================

Description:
Package initialization for the diabetic_retinopathy project. Exposes a small, stable
public API and version metadata.

Usage:
This file is imported implicitly when you import the package:

    import diabetic_retinopathy

Notes:
- Keep top-level imports minimal to avoid heavy side effects at import time.
"""

from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("diabetic-retinopathy-detector")
except PackageNotFoundError:  # pragma: no cover - during local development
    __version__ = "0.0.0"
