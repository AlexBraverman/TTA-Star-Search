import re


def extract_gsm8k_answer(text: str) -> str | None:
    """
    Extract the final numerical answer from model output.
    Tries '#### <number>' format first (standard GSM8K convention),
    then falls back to the last number in the text.
    """
    # Standard GSM8K / our solve-prompt format
    match = re.search(r"####\s*([-\d,\.]+)", text)
    if match:
        return _normalize(match.group(1))

    # Fallback: last standalone number
    numbers = re.findall(r"[-\d,\.]+", text)
    return _normalize(numbers[-1]) if numbers else None


def _normalize(s: str) -> str:
    """Strip commas, trailing zeros after decimal, etc."""
    s = s.replace(",", "").strip()
    try:
        val = float(s)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return s


def is_correct(predicted: str | None, ground_truth: str) -> bool:
    """
    Compare predicted answer to GSM8K ground truth.
    GSM8K ground truths end with '#### <number>'.
    """
    if predicted is None:
        return False
    gt = extract_gsm8k_answer(ground_truth) or ground_truth.strip()
    return predicted.strip() == gt.strip()
