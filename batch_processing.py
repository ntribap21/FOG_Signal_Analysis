"""Compatibility imports only. All algorithms live in signal_processing.py.
Replace old copies with this shim, or remove the module if no external code imports it.
"""
RELEASE_VERSION = 'fog-unified-1.4-cwt'
from signal_processing import *
LegacyEngine = ProcessingEngine
prepare_legacy_recording = prepare_recording
