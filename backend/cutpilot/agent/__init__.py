"""Planner — natural language + conversation state -> EditPlan.

The LLM never calls editing tools directly. It emits a declarative EditPlan
which the user reviews and edits; a local executor turns the approved plan
into tool calls.
"""
