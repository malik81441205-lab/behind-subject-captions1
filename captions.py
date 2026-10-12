"""SRT -> timed words -> display chunks (the unit that appears/disappears on screen)."""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Word:
    text: str
    start: float
    end: float
    is_keyword: bool = False


@dataclass
class Chunk:
    words: List[Word] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0


def _ts(s: str) -> float:
    h, m, rest = s.strip().split(":")
    sec, ms = rest.replace(".", ",").split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def parse_srt(path: str):
    """Returns [(start, end, text), ...]"""
    raw = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [l for l in block.split("\n") if l.strip()]
        for i, l in enumerate(lines):
            if "-->" in l:
                a, b = l.split("-->")
                text = " ".join(lines[i + 1:]).strip()
                if text:
                    cues.append((_ts(a), _ts(b), text))
                break
    return cues


def _clean(w: str) -> str:
    return re.sub(r"[^\w']", "", w.strip("*"))


STOPWORDS = set("""a an the and or but if so of to in on at by for with from as is are was were be been am
i you he she it we they me my your his her our their this that these those there here not no yes do does did
have has had will would can could should just very really then than too also about into over out up down""".split())


def _score(w: Word) -> int:
    c = _clean(w.text)
    return 0 if c.lower() in STOPWORDS else len(c)


def pick_keyword(words: List[Word]) -> None:
    """Explicit *marked* words win; otherwise the longest meaningful (non-filler) word in the chunk."""
    marked = [w for w in words if w.text.startswith("*") and w.text.endswith("*") and len(w.text) > 2]
    if marked:
        for w in marked:
            w.is_keyword = True
    else:
        best = max(words, key=lambda w: (_score(w), len(_clean(w.text))))
        best.is_keyword = True
    for w in words:
        w.text = w.text.strip("*")


def build_chunks(cues, words_per_chunk: int = 4) -> List[Chunk]:
    """Word timing is spread across each cue by character count. If you have real word timestamps
    (e.g. Whisper word-level output), build `Word` objects from them and reuse the chunking below."""
    chunks: List[Chunk] = []
    for start, end, text in cues:
        toks = text.split()
        if not toks:
            continue
        total = sum(len(_clean(t)) + 1 for t in toks)
        t, words = start, []
        for tok in toks:
            dur = (end - start) * (len(_clean(tok)) + 1) / total
            words.append(Word(tok, t, t + dur))
            t += dur
        for i in range(0, len(words), words_per_chunk):
            grp = words[i:i + words_per_chunk]
            pick_keyword(grp)
            chunks.append(Chunk(grp, grp[0].start, grp[-1].end))
    return chunks


def active_chunk(chunks: List[Chunk], t: float, hold: float = 0.0):
    for c in chunks:
        if c.start <= t < c.end + hold:
            return c
    return None


# ---------------- automatic captions: real word timings -> chunks ----------------
SENTENCE_END = (".", "?", "!", "…")


def chunk_words(word_dicts, words_per_chunk=4, pause=0.45, strip_punct=True, linger=0.35) -> List[Chunk]:
    """Group timed words into on-screen captions. A new caption starts at the end of a sentence,
    after a pause in the speech, or when the word limit is reached - so captions follow the voice."""
    words = [Word(d["text"], d["start"], d["end"]) for d in word_dicts]
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if (nxt is None or len(cur) >= words_per_chunk or w.text.endswith(SENTENCE_END)
                or nxt.start - w.end > pause):
            groups.append(cur); cur = []
    chunks = []
    for g in groups:
        pick_keyword(g)  # also removes *markers*
        if strip_punct:
            for w in g:
                w.text = w.text.strip(".,;:!?\"'\u201c\u201d\u2018\u2019()[]") or w.text
        chunks.append(Chunk(g, g[0].start, g[-1].end))
    for a, b in zip(chunks, chunks[1:]):          # keep a caption up briefly if the next one follows soon
        a.end = min(b.start, a.end + linger)
    if chunks:
        chunks[-1].end += linger
    return chunks


def words_to_srt(word_dicts, path, max_words=7, pause=0.6):
    """Optional: save the auto transcript as a normal .srt file."""
    def ts(t):
        h, m, s_ = int(t // 3600), int(t % 3600 // 60), t % 60
        return f"{h:02d}:{m:02d}:{int(s_):02d},{int(round((s_ % 1) * 1000)):03d}"
    lines, cur, idx = [], [], 1
    for i, d in enumerate(word_dicts):
        cur.append(d)
        nxt = word_dicts[i + 1] if i + 1 < len(word_dicts) else None
        if nxt is None or len(cur) >= max_words or d["text"].endswith(SENTENCE_END) or nxt["start"] - d["end"] > pause:
            lines.append(f"{idx}\n{ts(cur[0]['start'])} --> {ts(cur[-1]['end'])}\n{' '.join(x['text'] for x in cur)}\n")
            idx += 1; cur = []
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def load_chunks(video, srt=None, words_per_chunk=4, strip_punct=True, model="base", language=None):
    """The one entry point: use an .srt if given, otherwise listen to the video and time words to the voice."""
    if srt:
        return build_chunks(parse_srt(srt), words_per_chunk)
    from transcribe import transcribe
    return chunk_words(transcribe(video, model, language), words_per_chunk, strip_punct=strip_punct)
