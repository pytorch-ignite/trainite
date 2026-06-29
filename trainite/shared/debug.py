from enum import Flag, auto


class DebugFlag(Flag):
    NONE = 0
    GRADS = auto()  # Log gradient norms
    LOGITS = auto()  # Log logits statistics (mean, std, NaN check)
    LR = auto()  # Log current learning rate
    DATA = auto()  # Log batch shapes and padding stats
    ALL = GRADS | LOGITS | LR | DATA


def parse_debug_flags(flag_str: str | None) -> DebugFlag:
    """Parse 'GRADS|LR|LOGITS' or 'ALL' or 'NONE' into DebugFlag."""
    if not flag_str:
        return DebugFlag.NONE

    # Strip whitespace and split by '|'
    parts = [p.strip().upper() for p in flag_str.split("|")]

    result = DebugFlag.NONE
    for part in parts:
        if part == "NONE":
            continue
        elif part == "ALL":
            result |= DebugFlag.ALL
        elif hasattr(DebugFlag, part):
            result |= getattr(DebugFlag, part)
        else:
            raise ValueError(f"Unknown debug flag: {part}. Supported flags: GRADS, LOGITS, LR, DATA, ALL, NONE")

    return result
