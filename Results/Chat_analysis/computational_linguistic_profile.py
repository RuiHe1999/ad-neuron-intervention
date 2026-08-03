# 1. packages
import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from wordfreq import word_frequency

tqdm.pandas()

# 2. constants
DEFAULT_OUT_DIR = "automated_analysis"
DEFAULT_SPACY_MODEL = "en_core_web_lg"
DEFAULT_MATTR_WINDOW = 25

# 3. functions
def prune_and_reindex(dep_list, drop_rels=("punct",)):
    """Remove selected dependency relations and reindex retained tokens."""
    toks = [
        (item["text"], item["id"], item["head"], item["deprel"])
        for item in dep_list
    ]

    forms = {token_id: word for word, token_id, _, _ in toks}
    heads = {token_id: head for _, token_id, head, _ in toks}
    rels = {token_id: relation for _, token_id, _, relation in toks}
    nodes = [token_id for _, token_id, _, _ in toks]

    children = defaultdict(list)
    for _, token_id, head, _ in toks:
        if head != 0:
            children[head].append(token_id)

    drop_set = {relation.lower().strip() for relation in drop_rels}
    to_drop = {
        token_id
        for token_id in nodes
        if rels[token_id].lower().strip() in drop_set
        and heads[token_id] != 0
    }

    for token_id in sorted(to_drop):
        head = heads.get(token_id, 0)

        for child in children.get(token_id, []):
            heads[child] = head
            if head != 0:
                children[head].append(child)

        children[token_id] = []

    kept = [token_id for token_id in nodes if token_id not in to_drop]
    if not kept:
        return []

    if not any(heads[token_id] == 0 for token_id in kept):
        kept_set = set(kept)
        pointed = {
            heads[token_id]
            for token_id in kept
            if heads[token_id] in kept_set and heads[token_id] != 0
        }
        possible_roots = kept_set - pointed
        heads[min(possible_roots) if possible_roots else min(kept)] = 0

    remap = {
        old_id: new_id
        for new_id, old_id in enumerate(kept, start=1)
    }

    output = []
    for old_id in kept:
        old_head = heads.get(old_id, 0)
        output.append(
            (
                forms[old_id],
                remap[old_id],
                0 if old_head == 0 else remap.get(old_head, 0),
                rels[old_id],
            )
        )

    return output


def moving_average_ttr(tokens, window_size=25):
    """Calculate moving-average type-token ratio."""
    n_tokens = len(tokens)

    if n_tokens == 0:
        return np.nan

    if n_tokens <= window_size:
        return len(set(tokens)) / n_tokens

    values = [
        len(set(tokens[start:start + window_size])) / window_size
        for start in range(n_tokens - window_size + 1)
    ]
    return float(np.mean(values))


def calculate_depid_r(tokens, n_tokens):
    """Calculate revised dependency-based propositional idea density."""
    proposition_relations = {
        "advcl",
        "advmod",
        "amod",
        "appos",
        "cc",
        "csubj",
        "csubjpass",
        "det",
        "neg",
        "npadvmod",
        "nsubj",
        "nsubjpass",
        "nummod",
        "poss",
        "predet",
        "preconj",
        "prep",
        "quantmod",
        "tmod",
        "vmod",
    }

    propositions = []

    for token in tokens:
        relation = token.dep_
        lemma = token.lemma_.lower()
        head_lemma = token.head.lemma_.lower()

        if relation not in proposition_relations:
            continue

        if relation == "det" and lemma in {"a", "an", "the"}:
            continue

        if relation == "nsubj" and lemma in {"it", "this"}:
            continue

        propositions.append((relation, lemma, head_lemma))

    if n_tokens == 0:
        return np.nan

    return len(set(propositions)) / n_tokens


def calculate_clause_ratio(doc):
    """Calculate the proportion of retained dependency nodes forming clauses."""
    dependencies = []

    for token in doc:
        dependencies.append(
            {
                "text": token.text,
                "id": token.i + 1,
                "head": 0 if token.dep_ == "ROOT" else token.head.i + 1,
                "deprel": token.dep_,
            }
        )

    dependency = prune_and_reindex(
        dependencies,
        drop_rels=("punct",),
    )

    if not dependency:
        return np.nan

    clause_relations = {
        "csubj",
        "ccomp",
        "xcomp",
        "advcl",
        "acl",
        "relcl",
        "acl:relcl",
        "advcl:relcl",
        "csubj:outer",
        "csubj:pass",
    }

    counts = Counter(
        relation
        for _, _, _, relation in dependency
        if relation in clause_relations
    )

    return sum(counts.values()) / len(dependency)


def extract_traditional_features(response, nlp, mattr_window=25):
    """Extract six interpretable linguistic measures from one response."""
    response = "" if pd.isna(response) else str(response).strip()

    if not response:
        return {
            "n_tokens": 0,
            "mattr": np.nan,
            "LexicalH": np.nan,
            "lex_density": np.nan,
            "DEPID_R": np.nan,
            "clause_ratio": np.nan,
        }

    doc = nlp(response)

    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    excluded_pos = {"PUNCT", "SPACE", "SYM", "X"}

    tokens = [
        token
        for token in doc
        if token.pos_ not in excluded_pos
        and not token.is_punct
        and not token.is_space
        and token.text.strip()
    ]

    token_strings = [token.text.lower() for token in tokens]
    content_tokens = [
        token
        for token in tokens
        if token.pos_ in content_pos
    ]

    n_tokens = len(tokens)

    if n_tokens > 0:
        frequencies = [
            max(word_frequency(token, "en"), 1e-9)
            for token in token_strings
        ]
        lexical_surprisal = -float(np.mean(np.log2(frequencies)))
        lexical_density = len(content_tokens) / n_tokens
    else:
        lexical_surprisal = np.nan
        lexical_density = np.nan

    return {
        "n_tokens": n_tokens,
        "mattr": moving_average_ttr(
            token_strings,
            window_size=mattr_window,
        ),
        "LexicalH": lexical_surprisal,
        "lex_density": lexical_density,
        "DEPID_R": calculate_depid_r(tokens, n_tokens),
        "clause_ratio": calculate_clause_ratio(doc),
    }


def run_analysis(
    task_name,
    in_dir,
    out_dir,
    text_col="bot",
    spacy_model=DEFAULT_SPACY_MODEL,
    mattr_window=DEFAULT_MATTR_WINDOW,
):
    """Read one task file, extract traditional metrics, and save the result."""
    input_path = Path(in_dir) / f"{task_name}.xlsx"
    output_path = Path(out_dir) / f"{task_name}_ling.xlsx"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading spaCy model: {spacy_model}")
    nlp = spacy.load(spacy_model)

    print(f"Reading: {input_path}")
    data = pd.read_excel(input_path)

    if text_col not in data.columns:
        raise KeyError(
            f"Text column '{text_col}' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    print(f"Calculating six traditional metrics for {len(data)} responses...")

    features = data[text_col].progress_apply(
        lambda text: pd.Series(
            extract_traditional_features(
                text,
                nlp=nlp,
                mattr_window=mattr_window,
            )
        )
    )

    output = pd.concat(
        [
            data.reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output.to_excel(output_path, index=False)

    print(f"Saved: {output_path}")
    return output


# 4. commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Calculate six traditional linguistic metrics: token count, "
            "MATTR, lexical surprisal, lexical density, DEPID-R, and clause ratio."
        )
    )

    parser.add_argument(
        "task_name",
        type=str,
        help="Task name without .xlsx, for example immediate_recall.",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="summary_screen",
        help="Directory containing the input Excel files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Directory for saving the output file.",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="bot",
        help="Name of the column containing response text.",
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default=DEFAULT_SPACY_MODEL,
        help="spaCy English model used for POS and dependency parsing.",
    )
    parser.add_argument(
        "--mattr_window",
        type=int,
        default=DEFAULT_MATTR_WINDOW,
        help="Window size used for MATTR.",
    )

    args = parser.parse_args()

    run_analysis(
        task_name=args.task_name,
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        text_col=args.text_col,
        spacy_model=args.spacy_model,
        mattr_window=args.mattr_window,
    )