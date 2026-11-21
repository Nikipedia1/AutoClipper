"""Tests for Pydantic validation of viral clips."""
import pytest
from pydantic import ValidationError
from curator import ViralClip, ClipSelection


def test_valid_clip():
    clip = ViralClip(
        clip_index=1, start_sec=10.0, end_sec=45.0,
        hook_reason="Strong opening claim", viral_score=85,
        title="Amazing tip", social_caption="Watch this!",
        hashtags=["#shorts", "#viral"],
        recommended_platforms=["youtube_shorts", "tiktok"],
    )
    assert clip.duration_sec == 35.0
    assert clip.hashtags[0].startswith("#")


def test_invalid_duration():
    with pytest.raises(ValidationError):
        ViralClip(
            clip_index=1, start_sec=50.0, end_sec=40.0,
            hook_reason="x", viral_score=50, title="t",
            social_caption="c", hashtags=["#a"],
            recommended_platforms=["tiktok"],
        )


def test_hashtag_normalization():
    clip = ViralClip(
        clip_index=1, start_sec=0, end_sec=30,
        hook_reason="hook", viral_score=70, title="title",
        social_caption="caption",
        hashtags=["viral", "#shorts", "  #TIKTOK  "],
        recommended_platforms=["tiktok"],
    )
    assert all(h.startswith("#") for h in clip.hashtags)


def test_clip_selection():
    data = {"clips": [{
        "clip_index": 1, "start_sec": 5, "end_sec": 40,
        "hook_reason": "hook", "viral_score": 90, "title": "Best tip",
        "social_caption": "Try this", "hashtags": ["#tip"],
        "recommended_platforms": ["youtube_shorts"],
    }]}
    selection = ClipSelection.model_validate(data)
    assert len(selection.clips) == 1
