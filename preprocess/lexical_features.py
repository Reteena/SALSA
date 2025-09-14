#!/usr/bin/env python3
"""
Outputs per-utterance rows with:
 - emb_0..emb_383  (all-MiniLM-L6-v2),
 - utt_len, has_filler, start_time_s, end_time_s,
   noun_count, verb_count, pronoun_count, adj_count,
   noun_ratio, verb_ratio, pronoun_ratio, adj_ratio,
   mean_dep_distance, subordinate_clause_count, total_dependencies,
 - semantic_coherence (cosine with previous utterance embedding; first utt = 0.0)
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import math
from collections import Counter

import torch
from sentence_transformers import SentenceTransformer

import pylangacq
from scipy.spatial.distance import cosine
import nltk

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
nltk.download("punkt", quiet=True)

# ---------------------------
# Configuration / feature names
# ---------------------------
EMB_DIM = 384
EMB_NAMES = [f"emb_{i}" for i in range(EMB_DIM)]

HANDCRAFTED_KEYS = [
    "utt_len",
    "has_filler",
    "start_time_s",
    "end_time_s",
    "noun_count",
    "verb_count",
    "pronoun_count",
    "adj_count",
    "noun_ratio",
    "verb_ratio",
    "pronoun_ratio",
    "adj_ratio",
    "mean_dep_distance",
    "subordinate_clause_count",
    "total_dependencies",
]

ALL_FEATURE_NAMES = EMB_NAMES + HANDCRAFTED_KEYS + ["semantic_coherence"]

# Fillers we want to keep and canonicalize
FILLERS_RAW = {"&uh", "&um", "uh", "um", "uhh", "erm", "mhm", "mmh", "er", "ah", "mm", "hmm"}
FILLER_TOKEN = "<FILLER>"

# ---------------------------
# Utilities
# ---------------------------
def safe_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

_punct_re = re.compile(r"^[^\w&<>]+|[^\w<>]+$")  # strip outer punctuation but keep & for fillers and <> markers

def canonical_clean_token(tok: str):
    """
    Clean a single token from pylangacq. Returns:
      - cleaned_token (string) or None if token should be dropped
    Rules:
      - Map filler patterns to <FILLER>
      - Remove CHAT bracketed tags like [+ exc], [//], (.), etc.
      - Strip leading/trailing punctuation (but keep internal apostrophes)
      - Lowercase the token
    """
    if not tok:
        return None
    t = tok.strip()
    # drop CHAT-specific annotations
    if t.startswith('[') and t.endswith(']'):
        return None
    if t.startswith('<') and t.endswith('>'):
        # tokens like <stool is> may appear; strip angle brackets but keep content
        t = t[1:-1].strip()
        if not t:
            return None

    # Map fillers
    tl = t.lower()
    if tl in FILLERS_RAW:
        return FILLER_TOKEN

    # Remove parentheses-only tokens like (.) or (laugh)
    if tl.startswith('(') and tl.endswith(')'):
        inner = tl[1:-1].strip()
        if inner == '.' or inner == '':
            return None
        # else try inner
        tl = inner

    # Strip leading/trailing punctuation except keep & for &uh (we already handled fillers)
    t_stripped = _punct_re.sub("", t)
    if not t_stripped:
        return None

    # normalize whitespace and lower
    out = t_stripped.lower()
    return out

# New: token cleaning for embeddings (keep punctuation)
def clean_token_for_embedding(tok: str):
    """
    Minimal cleaning for text fed to MiniLM embeddings:
      - Drop CHAT bracketed tags like [+ exc], [//]
      - Flatten angle-bracket groups like <stool is> -> 'stool is'
      - Drop parenthetical markers like (.)
      - Preserve punctuation (do NOT strip), lowercase for consistency
    Returns the cleaned token string or None to drop it.
    """
    if not tok:
        return None
    t = tok.strip()
    # Drop CHAT-specific annotations
    if t.startswith('[') and t.endswith(']'):
        return None
    if t.startswith('<') and t.endswith('>'):
        inner = t[1:-1].strip()
        if not inner:
            return None
        t = inner
    tl = t.lower()
    # Remove parentheses-only tokens like (.) or (laugh)
    if tl.startswith('(') and tl.endswith(')'):
        inner = tl[1:-1].strip()
        if inner == '.' or inner == '':
            return None
        tl = inner
    # Keep punctuation as-is (no stripping)
    return tl

# ---------------------------
# CHAT parsing helpers
# ---------------------------
def parse_chat_transcript(file_path: str):
    """
    Return list of Utterance-like dicts for participant utterances.
    Each item contains:
      - 'participant', 'tokens' (raw token strings), 'text_raw' (joined raw tokens),
      - 'mor' (string or None), 'gra' (string or None), 'time_marks' (tuple or None)
    """
    reader = pylangacq.read_chat(file_path)
    utts = []
    for utt in reader.utterances(participants="PAR"):
        tokens = [t.word for t in utt.tokens] if getattr(utt, "tokens", None) else []
        text_raw = " ".join(tokens).strip()
        mor = utt.tiers.get("%mor") if getattr(utt, "tiers", None) else None
        gra = utt.tiers.get("%gra") if getattr(utt, "tiers", None) else None
        time_marks = utt.time_marks if getattr(utt, "time_marks", None) else None
        utts.append({
            "participant": utt.participant,
            "tokens": tokens,
            "text_raw": text_raw,
            "mor": mor,
            "gra": gra,
            "time_marks": time_marks
        })
    return utts

def extract_tokens_from_mor(mor_line):
    """
    Parse a %mor line into (word, base_pos) pairs.
    %mor tokens like: 'det:art|the' or 'n|cookie'
    Returns list of (word, pos_prefix)
    """
    if not mor_line:
        return []
    pairs = []
    for item in mor_line.split():
        if "|" not in item:
            continue
        left, right = item.split("|", 1)
        pos = left.split(":", 1)[0]  # 'det:art' -> 'det'
        word = right
        pairs.append((word, pos))
    return pairs

def extract_dependencies_from_gra(gra_line):
    """
    Parse a %gra line into list of (dep_index, head_index, relation)
    Example item: '1|2|DET' -> (1,2,'DET')
    """
    if not gra_line:
        return []
    deps = []
    for item in gra_line.split():
        parts = item.split("|")
        if len(parts) < 3:
            continue
        try:
            dep_idx = int(parts[0])
            head_idx = int(parts[1])
            rel = parts[2]
            deps.append((dep_idx, head_idx, rel))
        except ValueError:
            continue
    return deps

# ---------------------------
# Handcrafted utterance features (uses CLEANED tokens)
# ---------------------------
def compute_utterance_handcrafted_from_clean(tokens_clean, mor, gra, time_marks):
    """
    tokens_clean: list of cleaned tokens (strings) where fillers are '<FILLER>' preserved.
    mor, gra: raw tier strings
    time_marks: tuple(ms_start, ms_end) or None
    Returns dict of HANDCRAFTED_KEYS
    """
    # utt_len: count tokens excluding any None
    utt_len = len(tokens_clean)

    # has_filler
    has_filler = 1 if any(t == FILLER_TOKEN for t in tokens_clean) else 0

    # times: convert time_marks (assumed in ms) to seconds
    if time_marks and len(time_marks) == 2 and time_marks[0] is not None and time_marks[1] is not None:
        start_s = time_marks[0] / 1000.0
        end_s = time_marks[1] / 1000.0
    else:
        start_s = 0.0
        end_s = 0.0

    # POS counts from mor (mor is pos|word tokens)
    mor_pairs = extract_tokens_from_mor(mor)
    pos_counts = Counter([pos for _, pos in mor_pairs])
    noun_count = pos_counts.get("n", 0)
    verb_count = pos_counts.get("v", 0)
    pronoun_count = pos_counts.get("pro", 0)
    adj_count = pos_counts.get("adj", 0)
    adv_count = pos_counts.get("adv", 0)
    # total_content includes pronouns so ratios are comparable
    total_content = sum(pos_counts.get(tag, 0) for tag in ["n", "v", "adj", "adv", "pro"])

    noun_ratio = noun_count / total_content if total_content else 0.0
    verb_ratio = verb_count / total_content if total_content else 0.0
    pronoun_ratio = pronoun_count / total_content if total_content else 0.0
    adj_ratio = adj_count / total_content if total_content else 0.0

    # syntactic features from gra
    deps = extract_dependencies_from_gra(gra)
    dep_distances = [abs(dep_idx - head_idx) for dep_idx, head_idx, _ in deps if head_idx != 0]
    mean_dep_distance = float(np.mean(dep_distances)) if dep_distances else 0.0

    subordinate_relations = {"CMOD", "CPRED", "CJCT", "COMP", "XCOMP", "CCOMP", "ADVCL", "ACLS"}
    subordinate_count = sum(1 for _, _, rel in deps if any(sub_rel in rel for sub_rel in subordinate_relations))
    total_dependencies = len(deps)

    return {
        "utt_len": utt_len,
        "has_filler": has_filler,
        "start_time_s": start_s,
        "end_time_s": end_s,
        "noun_count": noun_count,
        "verb_count": verb_count,
        "pronoun_count": pronoun_count,
        "adj_count": adj_count,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "pronoun_ratio": pronoun_ratio,
        "adj_ratio": adj_ratio,
        "mean_dep_distance": mean_dep_distance,
        "subordinate_clause_count": subordinate_count,
        "total_dependencies": total_dependencies,
    }

# ---------------------------
# Embedding model loader
# ---------------------------
def load_test_labels(metadata_path):
    """
    Load test labels from meta_data_test.txt
    Returns dict mapping recording_id to cognitive_decline label (0 or 1)
    """
    labels = {}
    if not Path(metadata_path).exists():
        print(f"Warning: {metadata_path} not found, using -1 for all labels")
        return labels
    
    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('ID'):
                continue  # Skip header
            parts = [p.strip() for p in line.split(';')]
            if len(parts) >= 4:
                recording_id = parts[0]
                try:
                    label = int(parts[3])  # Label is in 4th column (0-indexed as 3)
                    labels[recording_id] = label
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse label for {recording_id}")
    
    print(f"Loaded labels for {len(labels)} recordings from {metadata_path}")
    return labels

# ---------------------------
# Main processing: utterance-level with semantic coherence
# ---------------------------
def process_directory_utterance_level(input_dir, output_csv, label_mode="subfolders", batch_size=64, metadata_path=None):
    """
    Process .cha files to produce utterance-level CSV with semantic_coherence.
    label_mode:
      - 'subfolders': expects input_dir/cc and input_dir/cd (train)
      - 'single': process all .cha in input_dir (test) with labels from metadata_path
    """
    # Load MiniLM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    # Load test labels if provided
    test_labels = {}
    if metadata_path and label_mode == "single":
        test_labels = load_test_labels(metadata_path)
    
    rows = []
    recording_ids = []
    utterance_ids = []
    labels = []

    if label_mode == "subfolders":
        groups = [("cc", 0), ("cd", 1)]
        file_iter = []
        for folder, label in groups:
            folder_path = Path(input_dir) / folder
            if not folder_path.exists():
                print(f"Warning: {folder_path} does not exist; skipping")
                continue
            for fp in sorted(folder_path.glob("*.cha")):
                file_iter.append((fp, label))
    else:
        file_iter = [(fp, test_labels.get(fp.stem, -1)) for fp in sorted(Path(input_dir).glob("*.cha"))]

    for fp, label in file_iter:
        rec_id = fp.stem
        try:
            utts = parse_chat_transcript(str(fp))
            if not utts:
                continue

            # Build two parallel streams per utterance:
            # 1) embed_texts: punctuation-preserved tokens for MiniLM
            # 2) cleaned_tokens_per_utt: punctuation-stripped tokens for handcrafted features
            embed_texts = []
            cleaned_tokens_per_utt = []
            for utt in utts:
                raw_tokens = utt["tokens"] or []
                # For embeddings: keep punctuation
                embed_tokens = []
                # For handcrafted features: strip punctuation
                cleaned_tokens = []
                for rtok in raw_tokens:
                    # Embedding token
                    e = clean_token_for_embedding(rtok)
                    if e is not None:
                        embed_tokens.append(e)
                    # Handcrafted token
                    c = canonical_clean_token(rtok)
                    if c is not None:
                        cleaned_tokens.append(c)
                # Build text fed to MiniLM (punctuation preserved)
                embed_text = " ".join(embed_tokens) if embed_tokens else ""
                embed_texts.append(embed_text)
                cleaned_tokens_per_utt.append(cleaned_tokens)

            # Batch encode embedding texts (on device; uses GPU if available)
            embs = model.encode(embed_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

            # compute semantic coherence as cosine with previous embedding
            prev_emb = None
            for i, utt in enumerate(utts):
                emb = embs[i] if i < len(embs) else np.zeros(EMB_DIM)
                # handcrafted computed from cleaned tokens & tiers
                handcrafted = compute_utterance_handcrafted_from_clean(
                    cleaned_tokens_per_utt[i], utt.get("mor"), utt.get("gra"), utt.get("time_marks")
                )
                # semantic coherence: cos similarity with prev utt embedding
                if prev_emb is not None:
                    sem_coh = safe_cosine_sim(emb, prev_emb)
                else:
                    sem_coh = 0.0
                prev_emb = emb

                hv = [handcrafted[k] for k in HANDCRAFTED_KEYS]
                vec = np.concatenate([emb, np.array(hv, dtype=float), np.array([sem_coh], dtype=float)], axis=0)
                rows.append(vec)
                recording_ids.append(rec_id)
                utterance_ids.append(f"{rec_id}_{i}")
                labels.append(label if label is not None else -1)
            print(f"Processed {rec_id}: {len(utts)} utterances")

        except Exception as e:
            print(f"Error processing {fp}: {repr(e)}")

    if not rows:
        print("No features extracted.")
        return

    # Build DataFrame with deterministic columns
    df = pd.DataFrame(rows, columns=ALL_FEATURE_NAMES)
    df.insert(0, "recording_id", recording_ids)
    df.insert(1, "utterance_id", utterance_ids)
    df.insert(2, "cognitive_decline", labels)

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} utterance rows to {output_csv}")

if __name__ == "__main__":
    # train: expects train/cc and train/cd folders
    process_directory_utterance_level("ADReSS-IS2020-data/train/transcription", "lexical_features_train.csv", label_mode="subfolders")
    # test: all .cha in test folder with labels from meta_data_test.txt
    test_metadata = "ADReSS-IS2020-data/test/meta_data_test.txt"
    process_directory_utterance_level("ADReSS-IS2020-data/test/transcription", "lexical_features_test.csv", label_mode="single", metadata_path=test_metadata)
