"""
dice_types.py
-------------
Single source of truth for dice-type vocabularies, shared by
dice_tracker.py, webapp.py, and eval_harness.py.

Face names match the YOLO model class names exactly. The combined
27-class model emits these directly, which is how "auto" mode knows what
dice type is in the tray without being told.
"""

TYPE_FACES = {
    "block": ["pow", "push", "both_down", "player_down", "stumble"],
    "d6":    ["1pip", "2pip", "3pip", "4pip", "5pip", "6BB"],
    "d16":   [f"D16_{n}" for n in range(1, 17)],
}

CLASS_TO_TYPE = {face: t for t, faces in TYPE_FACES.items()
                 for face in faces}


def majority_type(labels: list[str]) -> str | None:
    """Dice type implied by a set of detected face labels (majority vote;
    None when nothing recognizable is present)."""
    votes: dict[str, int] = {}
    for lab in labels:
        t = CLASS_TO_TYPE.get(lab)
        if t:
            votes[t] = votes.get(t, 0) + 1
    return max(votes, key=votes.get) if votes else None
