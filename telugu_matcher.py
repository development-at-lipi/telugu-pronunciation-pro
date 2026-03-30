"""
Telugu pronunciation matching — compares recognised text against expected text.

The comparison runs 6 checks in priority order. First match wins.
Handles: exact match, phonetic equivalents, starts-with, containment,
partial match, and Levenshtein similarity.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import List


@dataclass
class MatchResult:
    # ── Core verdict ─────────────────────────────────────────────
    match: bool
    score: int              # 0–100 overall score
    reason: str             # human-readable reason

    # ── Match type ───────────────────────────────────────────────
    match_type: str = ""    # exact | phonetic | starts_with | contains
                            # partial | levenshtein | none

    # ── Individual check scores (all 0–100) ──────────────────────
    exact_score: int = 0           # 100 if exact, else 0
    phonetic_score: int = 0        # 85 if phonetic equiv matched, else 0
    starts_with_score: int = 0     # 80 if recognised starts with expected
    contains_score: int = 0        # score if expected inside recognised
    partial_score: int = 0         # score if recognised inside expected
    levenshtein_score: int = 0     # raw levenshtein similarity %

    # ── Character / grapheme analytics ──────────────────────────
    char_overlap_pct: int = 0      # shared chars / max-length chars
    grapheme_match_pct: int = 0    # matching graphemes / total expected graphemes
    graphemes_expected: int = 0    # number of grapheme clusters in expected
    graphemes_recognized: int = 0  # number of grapheme clusters in recognised

    # ── Normalised strings (for debugging) ──────────────────────
    expected_normalized: str = ""
    recognized_normalized: str = ""

    # ── Context flags ────────────────────────────────────────────
    is_short_text: bool = False    # True when expected ≤ 2 graphemes (single letter)
    phonetic_alternatives: int = 0 # how many phonetic alts exist for this letter


# ──────────────────────────────────────────────────────────────────
# Telugu letter sets
# ──────────────────────────────────────────────────────────────────

TELUGU_LETTERS = {
    "vowels": [
        "అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ",
        "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ", "అం", "అః",
    ],
    "consonants": [
        "క", "ఖ", "గ", "ఘ", "ఙ",
        "చ", "ఛ", "జ", "ఝ", "ఞ",
        "ట", "ఠ", "డ", "ఢ", "ణ",
        "త", "థ", "ద", "ధ", "న",
        "ప", "ఫ", "బ", "భ", "మ",
        "య", "ర", "ల", "వ", "శ",
        "ష", "స", "హ", "ళ", "క్ష", "ఱ",
    ],
    "words": [
        "నమస్కారం", "ధన్యవాదాలు", "నీళ్ళు", "అమ్మ", "నాన్న",
        "పుస్తకం", "బడి", "ఇల్లు", "చెట్టు", "పువ్వు",
        "సూర్యుడు", "చంద్రుడు", "నక్షత్రం", "ఆకాశం", "భూమి",
    ],
}


# ──────────────────────────────────────────────────────────────────
# Phonetic equivalents — what STT engines actually return
# for each Telugu letter.
#
# Covers three root causes:
#   1. Engine returns English romanisation ("a", "ka", "ma" …)
#   2. Short/long vowel confusion (ఇ↔ఈ, ఉ↔ఊ …)
#   3. Retroflex/aspirated consonant swaps
# ──────────────────────────────────────────────────────────────────

PHONETIC_EQUIVALENTS = {
    # ── VOWELS ─────────────────────────────────────────
    "అ":  ["a", "aa", "ah", "అమ్మ", "అక్క", "అది", "అన్", "అప్"],
    "ఆ":  ["aa", "ah", "aha", "a", "ఆహ్", "ఆమె"],
    "ఇ":  ["i", "e", "ee", "ఈ", "ఇక", "in"],
    "ఈ":  ["ee", "i", "ఇ"],
    "ఉ":  ["u", "oo", "ఊ", "un", "up"],
    "ఊ":  ["uu", "oo", "u", "ఉ"],
    "ఋ":  ["రు", "రి", "ర", "రూ", "ru", "ri", "r"],
    "ౠ":  ["రూ", "రీ", "రు", "రి", "ర", "rri", "rru"],
    "ఎ":  ["e", "a", "ae", "ఏ", "em"],
    "ఏ":  ["ee", "e", "ay", "hey", "ఎ", "yay"],
    "ఐ":  ["ai", "aye", "eye", "i", "ఏ", "ae"],
    "ఒ":  ["o", "oh", "ఓ", "ఒక"],
    "ఓ":  ["oh", "o", "ow", "ఒ"],
    "ఔ":  ["au", "ow", "ou", "aw", "ఆ", "aow"],
    "అం": ["అన్", "అమ్", "అన", "అమ", "am", "an"],
    "అః": ["అహ", "అ", "aha", "ah"],

    # ── CONSONANTS ─────────────────────────────────────
    "క":  ["ka", "ga", "కా", "కి", "కు", "kaa", "k"],
    "ఖ":  ["kha", "ka", "ఖా", "kh", "ga"],
    "గ":  ["ga", "gaa", "గా"],
    "ఘ":  ["gha", "ga", "ఘా", "gaa", "gh"],
    "ఙ":  ["nga", "na", "న", "ఞ", "న్గ", "ng"],
    "చ":  ["cha", "ca", "చా"],
    "ఛ":  ["cha", "chha", "ఛా"],
    "జ":  ["ja", "jaa", "జా"],
    "ఝ":  ["jha", "ja", "ఝా", "za", "jh"],
    "ఞ":  ["nya", "na", "ni", "న", "ని", "న్య", "gna"],
    "ట":  ["ta", "tha", "టా", "త", "t", "taa", "Ta"],
    "ఠ":  ["tha", "ta", "ఠా", "థ", "టా", "Tha", "th"],
    "డ":  ["da", "డా", "ta", "ద", "Da", "daa"],
    "ఢ":  ["dha", "da", "ఢా", "డా", "ధ", "Dha", "dh"],
    "ణ":  ["na", "నా", "న", "Na", "naa"],
    "త":  ["ta", "tha", "తా", "te", "da", "t", "taa"],
    "థ":  ["tha", "ta", "థా", "th", "taa"],
    "ద":  ["da", "daa", "దా"],
    "ధ":  ["dha", "da", "ధా", "ద", "dh", "dhaa"],
    "న":  ["na", "ణ", "నా", "naa"],
    "ప":  ["pa", "paa", "పా"],
    "ఫ":  ["pha", "fa", "ఫా", "pa", "f", "ph", "phaa"],
    "బ":  ["ba", "బా", "va", "b", "pa", "baa"],
    "భ":  ["bha", "ba", "bhaa", "భా", "బ", "bh"],
    "మ":  ["ma", "మా", "maa", "m", "me"],
    "య":  ["ya", "yaa", "యా"],
    "ర":  ["ra", "raa", "రా"],
    "ల":  ["la", "laa", "లా"],
    "వ":  ["va", "wa", "వా"],
    "శ":  ["sha", "sa", "శా"],
    "ష":  ["sha", "షా"],
    "స":  ["sa", "సా"],
    "హ":  ["ha", "haa", "హా"],
    "ళ":  ["la", "ళా", "La"],
    "ఱ":  ["ర", "రా", "ra", "raa"],
    "క్ష": ["క్ష", "కష", "క్షా", "ksha", "ksh"],
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalise Telugu text for comparison."""
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    for ch in (" ", ".", ",", "?", "!", "।", "\u200c", "\u200d"):
        text = text.replace(ch, "")
    return text


def _graphemes(text: str) -> List[str]:
    """Split text into Telugu grapheme clusters (approximation)."""
    COMBINING = set(range(0x0C3E, 0x0C57)) | {0x0C4D, 0x0C02, 0x0C03, 0x0C55, 0x0C56}
    clusters: List[str] = []
    cur = ""
    for ch in text:
        if cur and ord(ch) in COMBINING:
            cur += ch
        else:
            if cur:
                clusters.append(cur)
            cur = ch
    if cur:
        clusters.append(cur)
    return clusters


def _levenshtein_pct(s1: str, s2: str) -> int:
    """Levenshtein similarity as a percentage (0–100)."""
    if not s1 or not s2:
        return 0
    n, m = len(s1), len(s2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if s1[i - 1] == s2[j - 1] else 1),
            )
            prev = tmp
    return int((1 - dp[m] / max(n, m)) * 100)


# ──────────────────────────────────────────────────────────────────
# Main comparison
# ──────────────────────────────────────────────────────────────────

def _char_overlap_pct(s1: str, s2: str) -> int:
    """Percentage of characters from s1 that appear (in order) in s2."""
    if not s1 or not s2:
        return 0
    matches = 0
    idx = 0
    for ch in s1:
        while idx < len(s2) and s2[idx] != ch:
            idx += 1
        if idx < len(s2):
            matches += 1
            idx += 1
    return int(matches / max(len(s1), len(s2)) * 100)


def _grapheme_match_pct(exp_g: List[str], rec_g: List[str]) -> int:
    """How many graphemes of exp appear anywhere in rec (order-independent)."""
    if not exp_g:
        return 0
    rec_copy = list(rec_g)
    matched = 0
    for g in exp_g:
        if g in rec_copy:
            matched += 1
            rec_copy.remove(g)
    return int(matched / len(exp_g) * 100)


def compare(expected: str, recognized: str) -> MatchResult:
    """
    Compare expected Telugu text with STT output.
    Runs 6 checks in order; first match wins.
    Always computes full analytics regardless of which check passes.
    """
    exp = _normalize(expected)
    rec = _normalize(recognized)

    # ── Pre-compute all analytics ─────────────────────────────────
    exp_g   = _graphemes(exp)
    rec_g   = _graphemes(rec)
    is_short = len(exp_g) <= 2

    lev_score    = _levenshtein_pct(exp, rec) if rec else 0
    char_overlap = _char_overlap_pct(exp, rec) if rec else 0
    g_match_pct  = _grapheme_match_pct(exp_g, rec_g) if rec else 0
    phonetic_alts = len(PHONETIC_EQUIVALENTS.get(exp, []))

    base = dict(
        levenshtein_score    = lev_score,
        char_overlap_pct     = char_overlap,
        grapheme_match_pct   = g_match_pct,
        graphemes_expected   = len(exp_g),
        graphemes_recognized = len(rec_g),
        expected_normalized  = exp,
        recognized_normalized= rec,
        is_short_text        = is_short,
        phonetic_alternatives= phonetic_alts,
    )

    if not rec:
        return MatchResult(False, 0, "No speech detected", match_type="none", **base)

    # ── 1. Exact match ───────────────────────────────────────────
    if exp == rec:
        return MatchResult(True, 100, "Perfect match",
                           match_type="exact", exact_score=100, **base)

    # ── 2. Phonetic equivalents ──────────────────────────────────
    for alt in PHONETIC_EQUIVALENTS.get(exp, []):
        alt_n = _normalize(alt)
        if rec == alt_n or rec.startswith(alt_n):
            return MatchResult(True, 85, "Phonetically correct",
                               match_type="phonetic", phonetic_score=85, **base)

    # ── 3. Starts-with ───────────────────────────────────────────
    if is_short and rec.startswith(exp):
        return MatchResult(True, 80, "Good pronunciation (extra syllable detected)",
                           match_type="starts_with", starts_with_score=80, **base)

    # ── 4. Expected contained in recognised ─────────────────────
    if exp in rec:
        ratio = len(exp) / len(rec)
        s = max(int(ratio * 100), 70)
        return MatchResult(True, s, "Good pronunciation",
                           match_type="contains", contains_score=s, **base)

    # ── 5. Recognised contained in expected ─────────────────────
    if rec in exp:
        ratio = len(rec) / len(exp)
        s = int(ratio * 100)
        threshold = 30 if len(exp) <= 3 else 50
        if s >= threshold:
            return MatchResult(True, s, "Partial match detected",
                               match_type="partial", partial_score=s, **base)

    # ── 6. Levenshtein ───────────────────────────────────────────
    threshold = 55 if len(exp) <= 3 else 60
    if lev_score >= threshold:
        return MatchResult(True, lev_score, "Close pronunciation",
                           match_type="levenshtein", **base)

    return MatchResult(False, lev_score,
                       f"Expected '{expected}', heard '{recognized}'",
                       match_type="none", **base)
