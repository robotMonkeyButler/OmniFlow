import argparse
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset


# --------------------------------------------------------------
# Text processing helpers
# --------------------------------------------------------------

# Dataset-specific instruction triggers: once any of these lines is encountered,
# we stop collecting further content and keep only the user utterance/transcript.
INSTRUCTION_TRIGGERS = {
    "cremad": [
        "the above is a speech recording along with the transcript",
    ],
    "meld_emotion": [
        "how would you describe the emotional state of the speaker",
    ],
    "meld_senti": [
        "the above is a video and audio recording from a conversation",
        "what is the sentiment of the speaker",
    ],
    "chsimsv2": [
        "given the following utterance from a segmented video clip",
    ],
    "ptsd_in_the_wild": [
        "above is a video along with the transcript of a patient undergoing clinical evaluation",
    ],
    "daicwoz": [
        "based on the participant's responses and behavior during this clinical interview",
    ],
    "tess": [
        "above is the transcript and an audio recording from a clinical context",
    ],
}


def _is_tag_line(line: str) -> bool:
    return bool(re.match(r"^<[^>]+>\s*$", line.strip()))


def clean_problem(problem: str, dataset_id: str = "") -> str:
    """Keep only the actual utterance/transcript, drop tags and instruction lines."""
    s = problem or ""
    lines = [ln.strip() for ln in s.splitlines()]
    triggers = [t.lower() for t in INSTRUCTION_TRIGGERS.get(dataset_id, [])]

    kept: List[str] = []
    for ln in lines:
        if not ln:
            continue
        if _is_tag_line(ln):
            # skip <video>/<audio>/<text> markers
            continue
        lnl = ln.lower()
        # stop collecting when hitting any instruction trigger
        if any(t in lnl for t in triggers):
            break
        kept.append(ln)

    if not kept:
        # fallback: remove tags via regex and keep entire text
        s2 = re.sub(r"^<[^>]+>\s*", "", s).strip()
        return re.sub(r"\s+", " ", s2).strip()

    merged = " ".join(kept)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase + keep alnum and apostrophes."""
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def load_glove(glove_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load GloVe txt file into memory and return (dict, unk_vector)."""
    embeddings: Dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word, values = parts[0], parts[1:]
            vec = np.asarray(values, dtype=np.float32)
            embeddings[word] = vec
    if not embeddings:
        raise ValueError(f"No embeddings loaded from {glove_path}")
    dim = len(next(iter(embeddings.values())))
    unk = np.zeros(dim, dtype=np.float32)
    return embeddings, unk


def encode_tokens(
    tokens: List[str], glove: Dict[str, np.ndarray], unk: np.ndarray
) -> np.ndarray:
    """Convert tokens to an array of shape (len, dim)."""
    vecs = [glove.get(tok, unk) for tok in tokens]
    if not vecs:
        vecs = [unk]
    return np.stack(vecs, axis=0).astype(np.float32)


# --------------------------------------------------------------
# Streaming statistics helpers
# --------------------------------------------------------------


class StreamingStats:
    """Incremental mean/std/max tracker for variable-length sequences."""

    def __init__(self) -> None:
        self.sum: Optional[torch.Tensor] = None
        self.sumsq: Optional[torch.Tensor] = None
        self.count: Optional[torch.Tensor] = None
        self.max_abs: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        """Accumulate statistics from a (N, D) tensor."""
        if values.ndim == 1:
            values = values.unsqueeze(0)

        flat = values.reshape(-1, values.shape[-1]).float()
        if flat.numel() == 0:
            return

        if self.sum is None:
            dim = flat.shape[-1]
            device = flat.device
            self.sum = torch.zeros(dim, dtype=torch.float32, device=device)
            self.sumsq = torch.zeros(dim, dtype=torch.float32, device=device)
            self.count = torch.zeros(dim, dtype=torch.float32, device=device)

        safe = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        valid = torch.isfinite(flat).float()

        self.sum = self.sum + (safe * valid).sum(dim=0)
        self.sumsq = self.sumsq + ((safe ** 2) * valid).sum(dim=0)
        self.count = self.count + valid.sum(dim=0)

        if valid.any():
            finite_vals = torch.abs(flat[torch.isfinite(flat)])
            if finite_vals.numel() > 0:
                self.max_abs = max(self.max_abs, finite_vals.max().item())

    def finalize(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        if self.sum is None or self.count is None:
            return None

        counts = torch.clamp(self.count, min=1.0)
        mean = self.sum / counts
        var = self.sumsq / counts - mean ** 2
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var).clamp_min(1e-6)
        clip = self.max_abs if self.max_abs > 0 else 1.0
        return mean.cpu(), std.cpu(), float(clip)


def _resolve_feature_path(root: str, rel_path: str) -> Optional[str]:
    if not rel_path:
        return None
    if os.path.isabs(rel_path):
        return rel_path if os.path.isfile(rel_path) else None
    full_path = os.path.join(root, rel_path)
    return full_path if os.path.isfile(full_path) else None


def compute_feature_stats_from_paths(
    feature_paths: List[str],
    feature_root: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
    stats = StreamingStats()
    for rel_path in feature_paths:
        full_path = _resolve_feature_path(feature_root, rel_path)
        if not full_path:
            continue
        try:
            feat = torch.load(full_path, map_location="cpu")
        except Exception:
            continue
        if not torch.is_tensor(feat):
            try:
                feat = torch.as_tensor(feat)
            except Exception:
                continue
        stats.update(feat.float())

    return stats.finalize()


def compute_text_feature_stats(text_embeddings: List[np.ndarray]) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
    stats = StreamingStats()
    for emb in text_embeddings:
        if emb is None:
            continue
        if not isinstance(emb, np.ndarray):
            emb = np.asarray(emb, dtype=np.float32)
        arr = torch.from_numpy(emb).float()
        stats.update(arr)
    return stats.finalize()


# --------------------------------------------------------------
# Core preprocessing
# --------------------------------------------------------------


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_outputs(
    output_dir: str,
    data: Dict[str, Dict[str, Any]],
    label_map: Dict[str, int],
    debug_metadata: Dict[str, List[Dict[str, Any]]],
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Save unified data structure with all splits under one directory.

    Args:
        output_dir: e.g., preprocessed_data/cremad/
        data: {"train": {...}, "valid": {...}, "test": {...}}
        label_map: {"happy": 0, "sad": 1, ...}
    """
    data_path = os.path.join(output_dir, "data.pkl")
    label_map_path = os.path.join(output_dir, "label_map.json")
    debug_meta_path = os.path.join(output_dir, "debug_metadata.json")
    stats_path = os.path.join(output_dir, "stats.pt")

    with open(data_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    with open(debug_meta_path, "w", encoding="utf-8") as f:
        json.dump(debug_metadata, f, ensure_ascii=False, indent=2)

    if stats is not None:
        torch.save(stats, stats_path)


def preprocess_dataset(
    dataset_id: str,
    glove_path: str,
    output_root: str,
    feature_root: str,
    text_policy: str = "problem",
) -> None:
    """Process all splits (train, valid, test) for a single dataset.

    Outputs to {output_root}/{dataset_id}/ with structure:
    {
        "data.pkl": {
            "train": {
                "text_embeddings": [...],
                "video_embed_paths": [...],
                "audio_embed_paths": [...],
                "labels": [...]
            },
            "valid": {...},
            "test": {...}
        },
        "label_map.json": {"happy": 0, "sad": 1, ...}
    }
    """
    if text_policy not in {"problem", "texts", "both"}:
        raise ValueError("text_policy must be one of ['problem','texts','both']")

    if not feature_root:
        raise ValueError("feature_root is required to compute modality statistics.")

    output_dir = os.path.join(output_root, dataset_id)
    ensure_dir(output_dir)

    glove, unk = load_glove(glove_path)

    def first_path(field: Any) -> str:
        if isinstance(field, list) and field:
            p0 = field[0]
            return p0 if isinstance(p0, str) else ""
        return ""

    def process_split(split: str) -> Dict[str, Any]:
        """Process a single split and return its data dict."""
        hf_split = "validation" if split == "valid" else split

        ds = load_dataset("keentomato/human_behavior_atlas", split=hf_split)
        ds = ds.filter(lambda x: x.get("dataset") == dataset_id)

        text_embeddings = []
        video_embed_paths = []
        audio_embed_paths = []
        labels = []
        debug_meta: List[Dict[str, Any]] = []

        for idx, row in enumerate(ds):
            problem_raw = row.get("problem", "")
            parts: List[str] = []
            if text_policy in ("problem", "both"):
                parts.append(clean_problem(problem_raw, dataset_id=dataset_id))
            if text_policy in ("texts", "both"):
                parts.append(row.get("texts", [""])[0] if row.get("texts") else "")
            merged = " ".join([p for p in parts if p]).strip()

            tokens = tokenize(merged)
            emb = encode_tokens(tokens, glove, unk)

            text_embeddings.append(emb)
            video_rel = first_path(row.get("ext_video_feats", []))
            audio_rel = first_path(row.get("ext_audio_feats", []))
            video_embed_paths.append(video_rel)
            audio_embed_paths.append(audio_rel)
            label_str = row.get("class_label", "")
            labels.append(label_str)

            # collect up to 10 metadata entries for debugging
            if len(debug_meta) < 10:
                debug_meta.append(
                    {
                        "id": idx,
                        "label_str": label_str,
                        "task": row.get("task", ""),
                        "dataset": dataset_id,
                        "problem_raw": problem_raw,
                        "problem_cleaned": merged,
                        "video_rel": video_rel,
                        "audio_rel": audio_rel,
                        "modality_signature": row.get("modality_signature", ""),
                    }
                )

        return {
            "text_embeddings": text_embeddings,
            "video_embed_paths": video_embed_paths,
            "audio_embed_paths": audio_embed_paths,
            "labels": labels,
            "debug_meta": debug_meta,
        }

    # Process all splits
    data = {
        "train": process_split("train"),
        "valid": process_split("valid"),
        "test": process_split("test"),
    }

    # Compute modality statistics from training split
    stats_payload: Dict[str, Any] = {"normalize": {}, "clip": {}}

    train_split = data["train"]

    vis_stats = compute_feature_stats_from_paths(
        train_split["video_embed_paths"], feature_root
    )
    if vis_stats is not None:
        mean, std, clip = vis_stats
        stats_payload["normalize"]["vis"] = (mean, std)
        stats_payload["clip"]["vis"] = clip

    aud_stats = compute_feature_stats_from_paths(
        train_split["audio_embed_paths"], feature_root
    )
    if aud_stats is not None:
        mean, std, clip = aud_stats
        stats_payload["normalize"]["aud"] = (mean, std)
        stats_payload["clip"]["aud"] = clip

    txt_stats = compute_text_feature_stats(train_split["text_embeddings"])
    if txt_stats is not None:
        mean, std, clip = txt_stats
        stats_payload["normalize"]["txt"] = (mean, std)
        stats_payload["clip"]["txt"] = clip

    has_stats = bool(stats_payload["normalize"] or stats_payload["clip"])
    final_stats = stats_payload if has_stats else None

    # Build unified label_map from all labels across all splits
    all_labels = set()
    for split_data in data.values():
        all_labels.update(split_data["labels"])
    label_map = {lb: i for i, lb in enumerate(sorted(all_labels))}

    # Assemble debug metadata per split
    debug_metadata = {split: data[split]["debug_meta"] for split in ["train", "valid", "test"]}

    save_outputs(
        output_dir=output_dir,
        data=data,
        label_map=label_map,
        debug_metadata=debug_metadata,
        stats=final_stats,
    )

    # Print summary
    total_samples = sum(len(split_data["labels"]) for split_data in data.values())
    print(f"✓ Preprocessed {dataset_id}: {total_samples} samples across 3 splits")
    print(f"  Output: {output_dir}")
    print(f"  Labels ({len(label_map)}): {', '.join(sorted(all_labels))}")
    if has_stats:
        print(f"  Stats: {os.path.join(output_dir, 'stats.pt')}")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess HBA sub-dataset with GloVe text features."
    )
    parser.add_argument(
        "--dataset_id", required=True, help="Sub-dataset id, e.g., cremad"
    )
    parser.add_argument("--glove_path", required=True, help="Path to glove.6B.300d.txt")
    parser.add_argument(
        "--output_dir",
        default="preprocessed_data",
        help="Where to save preprocessed outputs",
    )
    parser.add_argument(
        "--feature_root",
        required=True,
        help="Root directory containing pose/ and opensmile feature trees",
    )
    parser.add_argument(
        "--text_policy",
        default="problem",
        choices=["problem", "texts", "both"],
        help="Which text fields to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_dataset(
        dataset_id=args.dataset_id,
        glove_path=args.glove_path,
        output_root=args.output_dir,
        feature_root=args.feature_root,
        text_policy=args.text_policy,
    )


if __name__ == "__main__":
    main()
