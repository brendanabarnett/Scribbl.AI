from src.config import GAP_MULTIPLIER

def assemble_text(letters, centers):
    pairs = sorted(zip(centers, letters), key=lambda x: x[0])
    xs = [x for x, _ in pairs]
    chars = [c for _, c in pairs]
    if len(xs) > 1:
        diffs = [b - a for a, b in zip(xs, xs[1:])]
        gap = (sum(diffs) / len(diffs)) * GAP_MULTIPLIER
    else:
        gap = float('inf')
    text = ''
    prev = None
    for x, c in zip(xs, chars):
        if prev is not None and x - prev > gap:
            text += ' '
        text += c
        prev = x
    return text