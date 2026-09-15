"""Ingest pipeline — proxy transcode, thumbnail sprite, probe metadata.

Every upload produces three artifacts so the timeline UI can render without
touching the original file:
    1. proxy video   (low-res, fast scrubbing)
    2. sprite sheet  (timeline filmstrip)
    3. metadata      (duration, fps, resolution)
"""
