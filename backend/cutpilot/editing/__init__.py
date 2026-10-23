"""Executor -- applies an approved EditPlan: reframe, blur/mosaic, render.

The agent never reaches in here. A plan arrives as data, already grounded and
already approved, and nothing in this package asks a model anything, so the
same plan over the same video yields the same file every time.

    effects   blur / mosaic on a region
    reframe   crop-window geometry and the smoothed subject trajectory
    encoder   frames out to H.264 via an ffmpeg pipe, audio preserved
    executor  puts the three together over a video
"""

from cutpilot.editing.executor import RenderResult, execute

__all__ = ["execute", "RenderResult"]
