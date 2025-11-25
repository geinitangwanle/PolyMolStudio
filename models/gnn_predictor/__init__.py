"""
GNN predictor module wrapper.

Thin wrappers around the existing polyGeoGAT implementation so we can plug
it into a multi-module monorepo layout.
"""

from .model import GeoGATModel

__all__ = ["GeoGATModel"]
