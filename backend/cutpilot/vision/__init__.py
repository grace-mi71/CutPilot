"""Perception: detection (person/face/plate), tracking, and grounding.

Grounding sits behind a provider interface so the Gemini-backed implementation
can be swapped for a local VLM baseline during evaluation.
"""
