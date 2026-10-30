"""Paths to experiment artifacts and shared inputs in the parent repository."""
from notebook_api import ROOT as PACKAGE_ROOT
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

__all__ = ['PACKAGE_ROOT', 'REPOSITORY_ROOT', 'artifact_relative', 'resource_path']
