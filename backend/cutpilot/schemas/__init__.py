"""Pydantic domain models shared across layers.

These are the contract between backend and the (deferred) timeline UI:
    VideoAsset  uploaded video + proxy + thumbnail sprite + metadata
    Track       one tracked object with its presence segments on the timeline
    EditPlan    declarative, user-editable plan produced by the agent
"""
