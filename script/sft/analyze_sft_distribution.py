#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze SFT dataset distribution with lightweight text embeddings")
    parser.add_argument("--input-file", required=True, help="Filtered SFT jsonl file")
    parser.add_argument(
        "--output-root",
        default="/home/qjh/llm_learning/my_medical_gpt/data/sft/curation",
        help="Output root for analysis artifacts",
    )
    parser.add_argument("--output-name", default="medical_sft_27w_rule_filtered")
    parser.add_argument("--sample-size", type=int, default=30000, help="Max records used for vectorization")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--n-clusters", type=int, default=24)
    parser.add_argument("--representatives-per-cluster", type=int, default=5)
    return parser


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_text_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    conversations = record["conversations"]
    user_text = normalize_text(conversations[-2]["value"]) if len(conversations) >= 2 else ""
    assistant_text = normalize_text(conversations[-1]["value"]) if len(conversations) >= 1 else ""
    full_text = f"问题：{user_text}\n回答：{assistant_text}"
    return {
        "source": record.get("curation_source", "unknown"),
        "user_text": user_text,
        "assistant_text": assistant_text,
        "full_text": full_text,
        "turns": len(conversations),
        "user_chars": len(user_text),
        "assistant_chars": len(assistant_text),
    }


def choose_sample(rows: List[Dict[str, Any]], sample_size: int, random_seed: int) -> List[Dict[str, Any]]:
    if len(rows) <= sample_size:
        return rows
    rng = random.Random(random_seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = sorted(indices[:sample_size])
    return [rows[index] for index in chosen]


def safe_svd_components(matrix_shape: tuple[int, int], target_dim: int) -> int:
    upper = max(2, min(matrix_shape[0] - 1, matrix_shape[1] - 1, target_dim))
    return upper


def top_cluster_shares(cluster_counts: Dict[int, int], sample_size: int, top_k: int) -> float:
    if sample_size <= 0:
        return 0.0
    counts = sorted(cluster_counts.values(), reverse=True)
    return round(sum(counts[:top_k]) / sample_size, 4)


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input_file).resolve()
    output_root = Path(args.output_root).resolve()
    report_output = output_root / "analysis" / f"{args.output_name}.distribution.summary.json"
    cluster_output = output_root / "analysis" / f"{args.output_name}.cluster_stats.json"
    sample_output = output_root / "analysis" / f"{args.output_name}.sample_points.jsonl"

    raw_rows = read_jsonl(input_path)
    payload_rows = [build_text_payload(row) for row in raw_rows]
    sampled_rows = choose_sample(payload_rows, args.sample_size, args.random_seed)

    texts = [row["full_text"] for row in sampled_rows]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=args.max_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    sparse_matrix = vectorizer.fit_transform(texts)

    svd_dim = safe_svd_components(sparse_matrix.shape, args.svd_dim)
    svd = TruncatedSVD(n_components=svd_dim, random_state=args.random_seed)
    embeddings = svd.fit_transform(sparse_matrix)

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.random_seed,
        batch_size=2048,
        n_init="auto",
    )
    cluster_ids = kmeans.fit_predict(embeddings)

    pca = PCA(n_components=2, random_state=args.random_seed)
    points_2d = pca.fit_transform(embeddings)

    cluster_counts: Counter[int] = Counter(int(item) for item in cluster_ids)
    source_counts = Counter(row["source"] for row in sampled_rows)

    cluster_source_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    cluster_question_chars: Dict[int, List[int]] = defaultdict(list)
    cluster_answer_chars: Dict[int, List[int]] = defaultdict(list)
    for index, row in enumerate(sampled_rows):
        cluster_id = int(cluster_ids[index])
        cluster_source_counts[cluster_id][row["source"]] += 1
        cluster_question_chars[cluster_id].append(row["user_chars"])
        cluster_answer_chars[cluster_id].append(row["assistant_chars"])

    distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[cluster_ids], axis=1)
    cluster_representatives: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for cluster_id in range(args.n_clusters):
        member_indices = [index for index, current in enumerate(cluster_ids) if int(current) == cluster_id]
        nearest = sorted(member_indices, key=lambda idx: distances[idx])[: args.representatives_per_cluster]
        for idx in nearest:
            cluster_representatives[cluster_id].append(
                {
                    "source": sampled_rows[idx]["source"],
                    "user_text": sampled_rows[idx]["user_text"][:240],
                    "assistant_text": sampled_rows[idx]["assistant_text"][:320],
                    "distance_to_centroid": round(float(distances[idx]), 4),
                }
            )

    cluster_stats: List[Dict[str, Any]] = []
    for cluster_id, count in cluster_counts.most_common():
        source_distribution = {
            source: {
                "count": source_count,
                "ratio": round(source_count / count, 4),
            }
            for source, source_count in cluster_source_counts[cluster_id].most_common()
        }
        cluster_stats.append(
            {
                "cluster_id": cluster_id,
                "count": count,
                "ratio": round(count / len(sampled_rows), 4),
                "source_distribution": source_distribution,
                "avg_user_chars": round(sum(cluster_question_chars[cluster_id]) / len(cluster_question_chars[cluster_id]), 2),
                "avg_assistant_chars": round(sum(cluster_answer_chars[cluster_id]) / len(cluster_answer_chars[cluster_id]), 2),
                "representatives": cluster_representatives[cluster_id],
            }
        )

    largest_cluster_ratio = round(max(cluster_counts.values()) / len(sampled_rows), 4) if sampled_rows else 0.0
    top5_ratio = top_cluster_shares(dict(cluster_counts), len(sampled_rows), 5)
    source_entropy = 0.0
    for count in source_counts.values():
        p = count / len(sampled_rows)
        source_entropy -= p * math.log(p + 1e-12, 2)
    max_source_ratio = round(max(source_counts.values()) / len(sampled_rows), 4) if sampled_rows else 0.0

    recommendation_reasons: List[str] = []
    if largest_cluster_ratio >= 0.12:
        recommendation_reasons.append("largest_cluster_ratio_high")
    if top5_ratio >= 0.45:
        recommendation_reasons.append("top5_cluster_share_high")
    if max_source_ratio >= 0.78:
        recommendation_reasons.append("source_imbalance_high")

    recommendation = {
        "need_distribution_sampling": bool(recommendation_reasons),
        "need_theme_control": bool(recommendation_reasons),
        "reasons": recommendation_reasons,
        "suggestion": (
            "建议对大簇做限额抽样，并在 cluster/source 双维度上控制采样配额。"
            if recommendation_reasons
            else "当前分布未见特别极端的头部簇，可先不做强约束主题控制。"
        ),
    }

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "sample_size": len(sampled_rows),
        "total_rows": len(raw_rows),
        "vectorizer": {
            "type": "char_tfidf",
            "ngram_range": [2, 4],
            "max_features": args.max_features,
        },
        "embedding": {
            "type": "tfidf_svd",
            "svd_dim": svd_dim,
            "explained_variance_ratio_sum": round(float(svd.explained_variance_ratio_.sum()), 4),
        },
        "clustering": {
            "n_clusters": args.n_clusters,
            "largest_cluster_ratio": largest_cluster_ratio,
            "top5_cluster_ratio": top5_ratio,
        },
        "source_distribution": {
            source: {
                "count": count,
                "ratio": round(count / len(sampled_rows), 4),
            }
            for source, count in source_counts.most_common()
        },
        "source_entropy_bits": round(source_entropy, 4),
        "recommendation": recommendation,
    }

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cluster_output.write_text(json.dumps(cluster_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with sample_output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(sampled_rows):
            f.write(
                json.dumps(
                    {
                        "source": row["source"],
                        "cluster_id": int(cluster_ids[index]),
                        "x": round(float(points_2d[index, 0]), 6),
                        "y": round(float(points_2d[index, 1]), 6),
                        "user_chars": row["user_chars"],
                        "assistant_chars": row["assistant_chars"],
                        "user_text": row["user_text"][:160],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
