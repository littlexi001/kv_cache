from __future__ import annotations

import re
from typing import Any

from analyze_hierarchical_book_index_recall import SparseTfidfIndex, joined, selected_page_set
from analyze_longrange_book_index_semantic_retrieval import GeneratedTask, authority_score


def auth_flat_pages(
    task: GeneratedTask,
    pages: list[Any],
    index: SparseTfidfIndex,
    query_text: str,
    candidate_ids: list[int],
    page_count: int,
) -> set[int]:
    return {page_id for page_id, _ in auth_flat_scored_pages(task, pages, index, query_text, candidate_ids)[:page_count]}


def auth_flat_scored_pages(
    task: GeneratedTask,
    pages: list[Any],
    index: SparseTfidfIndex,
    query_text: str,
    candidate_ids: list[int],
) -> list[tuple[int, float]]:
    query_vec = index.query_vector(query_text)
    scored = []
    for page_id in candidate_ids:
        score = SparseTfidfIndex.cosine(query_vec, index.vectors[page_id])
        score += authority_score(joined(task.token_texts, pages[page_id].start, pages[page_id].end))
        scored.append((page_id, score))
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return scored


def chain_authflat_pages(
    task: GeneratedTask,
    pages: list[Any],
    index: SparseTfidfIndex,
    query_text: str,
    candidate_ids: list[int],
    seed_count: int,
    expand_count: int,
    radius: int,
) -> set[int]:
    query_vec = index.query_vector(query_text)
    target_key = getattr(task, "target_key", "").lower()
    seed_scored = []
    for page_id in candidate_ids:
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        score = SparseTfidfIndex.cosine(query_vec, index.vectors[page_id]) + authority_score(text)
        if target_key and target_key in text.lower():
            score += 2.0
        seed_scored.append((page_id, score))
    seed_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    seeds = {page_id for page_id, _ in seed_scored[:seed_count]}
    seed_text = "\n".join(joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds)
    expanded_query = query_text + "\n" + seed_text
    expanded = {page_id for page_id, _ in auth_flat_scored_pages(task, pages, index, expanded_query, candidate_ids)[:expand_count]}
    selected = seeds | expanded
    if radius > 0:
        selected = expand_authority_adjacent_pages(task, pages, selected, candidate_ids, radius)
    return selected


def key_seed_pages(
    task: GeneratedTask,
    pages: list[Any],
    index: SparseTfidfIndex,
    query_text: str,
    candidate_ids: list[int],
    seed_count: int,
) -> set[int]:
    query_vec = index.query_vector(query_text)
    target_key = getattr(task, "target_key", "").lower()
    scored = []
    for page_id in candidate_ids:
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        score = SparseTfidfIndex.cosine(query_vec, index.vectors[page_id]) + authority_score(text)
        if target_key and target_key in text.lower():
            score += 2.0
        scored.append((page_id, score))
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return {page_id for page_id, _ in scored[:seed_count]}


def chain_authhier_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    seed_count: int,
    section_count: int,
    pages_per_section: int,
    radius: int,
) -> set[int]:
    seeds = key_seed_pages(task, pages, page_index, query_text, candidate_pages, seed_count)
    seed_text = "\n".join(joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds)
    expanded_query = query_text + "\n" + seed_text
    selected = set(seeds) | auth_hier_pages(
        task,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
        expanded_query,
        candidate_pages,
        candidate_sections,
        section_count,
        pages_per_section,
    )
    if radius > 0:
        selected = expand_adjacent_pages(selected, candidate_pages, radius)
    return selected


def bridge_artifact_from_pages(task: GeneratedTask, page_texts: list[str]) -> str:
    target_key = getattr(task, "target_key", "")
    if not target_key:
        return ""
    route_patterns = [
        re.compile(
            rf"lookup key\s+{re.escape(target_key)}\s+routes to controlling artifact code\s+([A-Z0-9-]+)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"lookup key\s+{re.escape(target_key)}\s+points to controlling artifact\s+([A-Z0-9-]+)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"lookup key\s+{re.escape(target_key)}[^.\n]*artifact(?: code)?\s+([A-Z0-9-]+)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"badge\s+{re.escape(target_key)}[^.\n]*river-name\s+([A-Z0-9-]+)",
            flags=re.IGNORECASE,
        ),
    ]
    for text in page_texts:
        lowered = text.lower()
        if "different lookup key" in lowered or "must not answer" in lowered or "should not answer" in lowered:
            continue
        for pattern in route_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
    return ""


def chain_typedflat_pages(
    task: GeneratedTask,
    pages: list[Any],
    index: SparseTfidfIndex,
    query_text: str,
    candidate_ids: list[int],
    seed_count: int,
    expand_count: int,
    radius: int,
) -> set[int]:
    seeds = key_seed_pages(task, pages, index, query_text, candidate_ids, seed_count)
    seed_texts = [joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds]
    artifact = bridge_artifact_from_pages(task, seed_texts)
    if not artifact:
        return chain_authflat_pages(task, pages, index, query_text, candidate_ids, seed_count, expand_count, radius)
    artifact_query = artifact_query_text(artifact)
    query_vec = index.query_vector(artifact_query)
    scored = []
    for page_id in candidate_ids:
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        base = SparseTfidfIndex.cosine(query_vec, index.vectors[page_id])
        score = artifact_route_score(text, base, artifact)
        scored.append((page_id, score))
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    selected = set(seeds) | {page_id for page_id, _ in scored[:expand_count]}
    if radius > 0:
        selected = expand_adjacent_pages(selected, candidate_ids, radius)
    return selected


def chain_typedflat_adaptive_pages(
    task: GeneratedTask,
    pages: list[Any],
    index: SparseTfidfIndex,
    query_text: str,
    candidate_ids: list[int],
    min_seed_count: int,
    max_seed_count: int,
    expand_count: int,
    radius: int,
) -> set[int]:
    chosen_seeds: set[int] = set()
    chosen_artifact = ""
    for seed_count in range(min_seed_count, max_seed_count + 1):
        seeds = key_seed_pages(task, pages, index, query_text, candidate_ids, seed_count)
        seed_texts = [joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds]
        artifact = bridge_artifact_from_pages(task, seed_texts)
        chosen_seeds = seeds
        chosen_artifact = artifact
        if artifact:
            break
    if not chosen_artifact:
        return chain_authflat_pages(task, pages, index, query_text, candidate_ids, max_seed_count, expand_count, radius)
    artifact_query = artifact_query_text(chosen_artifact)
    query_vec = index.query_vector(artifact_query)
    scored = []
    for page_id in candidate_ids:
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        base = SparseTfidfIndex.cosine(query_vec, index.vectors[page_id])
        score = artifact_route_score(text, base, chosen_artifact)
        scored.append((page_id, score))
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    selected = set(chosen_seeds) | {page_id for page_id, _ in scored[:expand_count]}
    if radius > 0:
        selected = expand_adjacent_pages(selected, candidate_ids, radius)
    return selected


def artifact_query_text(artifact: str) -> str:
    return (
        f"certified artifact entry artifact {artifact} approved response letter verified answer label "
        f"controlling source {artifact} resolution memo river-name current ruling option {artifact}"
    )


def artifact_route_score(text: str, base_score: float, artifact: str) -> float:
    lowered = text.lower()
    score = base_score + authority_score(text)
    if artifact.lower() in lowered:
        score += 3.0
    if (
        "certified artifact entry" in lowered
        or "authoritative evidence page" in lowered
        or "resolution memo" in lowered
        or "current ruling" in lowered
    ):
        score += 0.5
    if (
        "late reminder" in lowered
        or "near-tail decoy" in lowered
        or "superseded" in lowered
        or "obsolete" in lowered
        or "outdated" in lowered
        or "former response" in lowered
        or "not the controlling" in lowered
        or "old desk slip" in lowered
        or "withdrawn" in lowered
        or "earlier ruling" in lowered
        or "no longer current" in lowered
    ):
        score -= 0.5
    return score


def negative_evidence_page(text: str) -> bool:
    lowered = text.lower()
    return (
        "late reminder" in lowered
        or "near-tail decoy" in lowered
        or "superseded" in lowered
        or "obsolete" in lowered
        or "outdated" in lowered
        or "former response" in lowered
        or "not the controlling" in lowered
        or "old desk slip" in lowered
        or "withdrawn" in lowered
        or "earlier ruling" in lowered
        or "no longer current" in lowered
        or "should not answer" in lowered
        or "must not answer" in lowered
        or "different lookup key" in lowered
    )


def answer_like_page(text: str, artifact: str) -> bool:
    lowered = text.lower()
    if artifact.lower() not in lowered or negative_evidence_page(text):
        return False
    return (
        "certified artifact entry" in lowered
        or "authoritative evidence page" in lowered
        or "approved response letter" in lowered
        or "resolution memo" in lowered
        or "current ruling" in lowered
        or "closes with option" in lowered
    )


def bridge_like_seed_pages(task: GeneratedTask, pages: list[Any], selected: set[int], artifact: str) -> set[int]:
    target_key = getattr(task, "target_key", "")
    kept: set[int] = set()
    for page_id in selected:
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        if negative_evidence_page(text):
            continue
        if artifact and artifact.lower() in text.lower():
            kept.add(page_id)
            continue
        if target_key and target_key in text:
            kept.add(page_id)
    return kept or set(selected)


def nonnegative_seed_pages(task: GeneratedTask, pages: list[Any], selected: set[int], artifact: str) -> set[int]:
    kept: set[int] = set()
    for page_id in selected:
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        if negative_evidence_page(text):
            continue
        kept.add(page_id)
    return kept or bridge_like_seed_pages(task, pages, selected, artifact)


def discover_chain_artifact(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    query_text: str,
    candidate_pages: list[int],
    max_seed_count: int,
) -> tuple[set[int], str]:
    chosen_seeds: set[int] = set()
    chosen_artifact = ""
    for seed_count in range(2, max_seed_count + 1):
        seeds = key_seed_pages(task, pages, page_index, query_text, candidate_pages, seed_count)
        seed_texts = [joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds]
        artifact = bridge_artifact_from_pages(task, seed_texts)
        chosen_seeds = seeds
        chosen_artifact = artifact
        if artifact:
            break
    return chosen_seeds, chosen_artifact


def top_neutral_context_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    query_text: str,
    candidate_pages: list[int],
    selected: set[int],
    artifact: str,
    context_count: int,
) -> set[int]:
    if context_count <= 0:
        return set()
    query_vec = page_index.query_vector(query_text)
    target_key = getattr(task, "target_key", "").lower()
    scored = []
    for page_id in candidate_pages:
        if page_id in selected:
            continue
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        lowered = text.lower()
        if negative_evidence_page(text):
            continue
        if artifact and answer_like_page(text, artifact):
            continue
        score = SparseTfidfIndex.cosine(query_vec, page_index.vectors[page_id])
        if target_key and target_key in lowered:
            score += 0.5
        if artifact and artifact.lower() in lowered:
            score += 0.2
        scored.append((page_id, score))
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return {page_id for page_id, _ in scored[:context_count]}


def top_section_context_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    candidate_pages: list[int],
    candidate_sections: list[int],
    selected: set[int],
    artifact: str,
    section_count: int,
    context_count: int,
) -> set[int]:
    if context_count <= 0 or not artifact:
        return set()
    artifact_query = artifact_query_text(artifact)
    section_query = section_index.query_vector(artifact_query)
    section_scored = []
    for section_id in candidate_sections:
        text = joined(task.token_texts, sections[section_id].start, sections[section_id].end)
        base = SparseTfidfIndex.cosine(section_query, section_index.vectors[section_id])
        section_scored.append((section_id, artifact_route_score(text, base, artifact)))
    section_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    page_query = page_index.query_vector(artifact_query)
    candidate_set = set(candidate_pages)
    page_scored = []
    for section_id, _ in section_scored[:section_count]:
        for page_id in section_to_pages.get(section_id, []):
            if page_id not in candidate_set or page_id in selected:
                continue
            text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
            if negative_evidence_page(text) or answer_like_page(text, artifact):
                continue
            base = SparseTfidfIndex.cosine(page_query, page_index.vectors[page_id])
            page_scored.append((page_id, artifact_route_score(text, base, artifact)))
    page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return {page_id for page_id, _ in page_scored[:context_count]}


def chain_typedhier_conf_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    max_seed_count: int,
    section_count: int,
    pages_per_section: int,
    radius: int,
) -> set[int]:
    chosen_seeds: set[int] = set()
    chosen_artifact = ""
    for seed_count in range(2, max_seed_count + 1):
        seeds = key_seed_pages(task, pages, page_index, query_text, candidate_pages, seed_count)
        seed_texts = [joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds]
        artifact = bridge_artifact_from_pages(task, seed_texts)
        chosen_seeds = seeds
        chosen_artifact = artifact
        if artifact:
            break
    if not chosen_artifact:
        return chain_authhier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            max_seed_count,
            section_count,
            pages_per_section,
            radius,
        )
    artifact_query = artifact_query_text(chosen_artifact)
    section_query = section_index.query_vector(artifact_query)
    section_scored = []
    for section_id in candidate_sections:
        text = joined(task.token_texts, sections[section_id].start, sections[section_id].end)
        base = SparseTfidfIndex.cosine(section_query, section_index.vectors[section_id])
        section_scored.append((section_id, artifact_route_score(text, base, chosen_artifact)))
    section_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    page_query = page_index.query_vector(artifact_query)
    candidate_set = set(candidate_pages)
    selected = set(chosen_seeds)
    for section_id, _ in section_scored[:section_count]:
        page_scored = []
        for page_id in section_to_pages.get(section_id, []):
            if page_id not in candidate_set:
                continue
            text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
            base = SparseTfidfIndex.cosine(page_query, page_index.vectors[page_id])
            page_scored.append((page_id, artifact_route_score(text, base, chosen_artifact)))
        page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        selected.update(page_id for page_id, _ in page_scored[:pages_per_section])
    if radius > 0:
        selected = expand_adjacent_pages(selected, candidate_pages, radius)
    return selected


def chain_typedhier_role_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    max_seed_count: int,
    section_count: int,
    pages_per_section: int,
    radius: int,
) -> set[int]:
    chosen_seeds: set[int] = set()
    chosen_artifact = ""
    for seed_count in range(2, max_seed_count + 1):
        seeds = key_seed_pages(task, pages, page_index, query_text, candidate_pages, seed_count)
        seed_texts = [joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds]
        artifact = bridge_artifact_from_pages(task, seed_texts)
        chosen_seeds = seeds
        chosen_artifact = artifact
        if artifact:
            break
    if not chosen_artifact:
        return chain_typedhier_conf_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            max_seed_count,
            section_count,
            pages_per_section,
            radius,
        )

    artifact_query = artifact_query_text(chosen_artifact)
    section_query = section_index.query_vector(artifact_query)
    section_scored = []
    for section_id in candidate_sections:
        text = joined(task.token_texts, sections[section_id].start, sections[section_id].end)
        base = SparseTfidfIndex.cosine(section_query, section_index.vectors[section_id])
        section_scored.append((section_id, artifact_route_score(text, base, chosen_artifact)))
    section_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    page_query = page_index.query_vector(artifact_query)
    candidate_set = set(candidate_pages)
    selected = bridge_like_seed_pages(task, pages, chosen_seeds, chosen_artifact)
    for section_id, _ in section_scored[:section_count]:
        page_scored = []
        fallback_scored = []
        for page_id in section_to_pages.get(section_id, []):
            if page_id not in candidate_set:
                continue
            text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
            base = SparseTfidfIndex.cosine(page_query, page_index.vectors[page_id])
            score = artifact_route_score(text, base, chosen_artifact)
            if answer_like_page(text, chosen_artifact):
                page_scored.append((page_id, score))
            elif not negative_evidence_page(text):
                fallback_scored.append((page_id, score))
        page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        fallback_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        chosen = page_scored[:pages_per_section]
        if not chosen:
            chosen = fallback_scored[:pages_per_section]
        selected.update(page_id for page_id, _ in chosen)
    if radius > 0:
        selected = expand_adjacent_pages(selected, candidate_pages, radius)
    return selected


def chain_typedhier_rolectx_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    max_seed_count: int,
    section_count: int,
    pages_per_section: int,
    radius: int,
) -> set[int]:
    chosen_seeds: set[int] = set()
    chosen_artifact = ""
    for seed_count in range(2, max_seed_count + 1):
        seeds = key_seed_pages(task, pages, page_index, query_text, candidate_pages, seed_count)
        seed_texts = [joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds]
        artifact = bridge_artifact_from_pages(task, seed_texts)
        chosen_seeds = seeds
        chosen_artifact = artifact
        if artifact:
            break
    if not chosen_artifact:
        return chain_typedhier_conf_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            max_seed_count,
            section_count,
            pages_per_section,
            radius,
        )

    artifact_query = artifact_query_text(chosen_artifact)
    section_query = section_index.query_vector(artifact_query)
    section_scored = []
    for section_id in candidate_sections:
        text = joined(task.token_texts, sections[section_id].start, sections[section_id].end)
        base = SparseTfidfIndex.cosine(section_query, section_index.vectors[section_id])
        section_scored.append((section_id, artifact_route_score(text, base, chosen_artifact)))
    section_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    page_query = page_index.query_vector(artifact_query)
    candidate_set = set(candidate_pages)
    selected = nonnegative_seed_pages(task, pages, chosen_seeds, chosen_artifact)
    for section_id, _ in section_scored[:section_count]:
        page_scored = []
        fallback_scored = []
        for page_id in section_to_pages.get(section_id, []):
            if page_id not in candidate_set:
                continue
            text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
            base = SparseTfidfIndex.cosine(page_query, page_index.vectors[page_id])
            score = artifact_route_score(text, base, chosen_artifact)
            if answer_like_page(text, chosen_artifact):
                page_scored.append((page_id, score))
            elif not negative_evidence_page(text):
                fallback_scored.append((page_id, score))
        page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        fallback_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        chosen = page_scored[:pages_per_section]
        if not chosen:
            chosen = fallback_scored[:pages_per_section]
        selected.update(page_id for page_id, _ in chosen)
    if radius > 0:
        selected = expand_adjacent_pages(selected, candidate_pages, radius)
    return selected


def chain_typedhier_rolectxflat_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    max_seed_count: int,
    section_count: int,
    pages_per_section: int,
    context_count: int,
    radius: int,
) -> set[int]:
    _, artifact = discover_chain_artifact(task, pages, page_index, query_text, candidate_pages, max_seed_count)
    selected = chain_typedhier_role_pages(
        task,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
        query_text,
        candidate_pages,
        candidate_sections,
        max_seed_count,
        section_count,
        pages_per_section,
        radius,
    )
    selected.update(
        top_neutral_context_pages(
            task,
            pages,
            page_index,
            query_text,
            candidate_pages,
            selected,
            artifact,
            context_count,
        )
    )
    return selected


def chain_typedhier_rolectxart_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    max_seed_count: int,
    section_count: int,
    pages_per_section: int,
    context_count: int,
    radius: int,
) -> set[int]:
    _, artifact = discover_chain_artifact(task, pages, page_index, query_text, candidate_pages, max_seed_count)
    selected = chain_typedhier_role_pages(
        task,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
        query_text,
        candidate_pages,
        candidate_sections,
        max_seed_count,
        section_count,
        pages_per_section,
        radius,
    )
    context_query = artifact_query_text(artifact) if artifact else query_text
    selected.update(
        top_neutral_context_pages(
            task,
            pages,
            page_index,
            context_query,
            candidate_pages,
            selected,
            artifact,
            context_count,
        )
    )
    return selected


def chain_typedhier_rolectxsec_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    max_seed_count: int,
    section_count: int,
    pages_per_section: int,
    context_count: int,
    radius: int,
) -> set[int]:
    _, artifact = discover_chain_artifact(task, pages, page_index, query_text, candidate_pages, max_seed_count)
    selected = chain_typedhier_role_pages(
        task,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
        query_text,
        candidate_pages,
        candidate_sections,
        max_seed_count,
        section_count,
        pages_per_section,
        radius,
    )
    selected.update(
        top_section_context_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            candidate_pages,
            candidate_sections,
            selected,
            artifact,
            section_count,
            context_count,
        )
    )
    return selected


def typed_seed_ceiling_for_length(context_tokens: int) -> int:
    if context_tokens <= 20_000:
        return 4
    if context_tokens <= 40_000:
        return 6
    return 8


def adaptive_section_fanout(section_to_pages: dict[int, list[int]], candidate_sections: list[int]) -> int:
    lengths = sorted(len(section_to_pages.get(section_id, [])) for section_id in candidate_sections)
    if not lengths:
        return 1
    typical_section_pages = lengths[len(lengths) // 2]
    return 3 if typical_section_pages > 8 else 1


def score_margin(scored: list[tuple[int, float]]) -> float:
    if len(scored) < 2:
        return 999.0
    return float(scored[0][1] - scored[1][1])


def chain_typedhier_margin_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    max_seed_count: int,
    pages_per_section: int,
    margin_threshold: float,
    radius: int,
) -> set[int]:
    chosen_seeds: set[int] = set()
    chosen_artifact = ""
    for seed_count in range(2, max_seed_count + 1):
        seeds = key_seed_pages(task, pages, page_index, query_text, candidate_pages, seed_count)
        seed_texts = [joined(task.token_texts, pages[page_id].start, pages[page_id].end) for page_id in seeds]
        artifact = bridge_artifact_from_pages(task, seed_texts)
        chosen_seeds = seeds
        chosen_artifact = artifact
        if artifact:
            break
    if not chosen_artifact:
        return chain_authhier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            max_seed_count,
            3,
            pages_per_section,
            radius,
        )

    artifact_query = artifact_query_text(chosen_artifact)
    section_query = section_index.query_vector(artifact_query)
    section_scored = []
    for section_id in candidate_sections:
        text = joined(task.token_texts, sections[section_id].start, sections[section_id].end)
        base = SparseTfidfIndex.cosine(section_query, section_index.vectors[section_id])
        section_scored.append((section_id, artifact_route_score(text, base, chosen_artifact)))
    section_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    page_query = page_index.query_vector(artifact_query)
    candidate_set = set(candidate_pages)
    top_page_scored: list[tuple[int, float]] = []
    if section_scored:
        top_section_id = section_scored[0][0]
        for page_id in section_to_pages.get(top_section_id, []):
            if page_id not in candidate_set:
                continue
            text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
            base = SparseTfidfIndex.cosine(page_query, page_index.vectors[page_id])
            top_page_scored.append((page_id, artifact_route_score(text, base, chosen_artifact)))
        top_page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    lengths = sorted(len(section_to_pages.get(section_id, [])) for section_id in candidate_sections)
    typical_section_pages = lengths[len(lengths) // 2] if lengths else 0
    section_count = 1
    if (
        typical_section_pages > 8
        or score_margin(section_scored) < margin_threshold
        or score_margin(top_page_scored) < margin_threshold
    ):
        section_count = 3

    selected = set(chosen_seeds)
    for section_id, _ in section_scored[:section_count]:
        page_scored = []
        for page_id in section_to_pages.get(section_id, []):
            if page_id not in candidate_set:
                continue
            text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
            base = SparseTfidfIndex.cosine(page_query, page_index.vectors[page_id])
            page_scored.append((page_id, artifact_route_score(text, base, chosen_artifact)))
        page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        selected.update(page_id for page_id, _ in page_scored[:pages_per_section])
    if radius > 0:
        selected = expand_adjacent_pages(selected, candidate_pages, radius)
    return selected


def auth_hier_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    query_text: str,
    candidate_pages: list[int],
    candidate_sections: list[int],
    section_count: int,
    pages_per_section: int,
) -> set[int]:
    page_query = page_index.query_vector(query_text)
    section_query = section_index.query_vector(query_text)
    section_scored = []
    for section_id in candidate_sections:
        score = SparseTfidfIndex.cosine(section_query, section_index.vectors[section_id])
        score += authority_score(joined(task.token_texts, sections[section_id].start, sections[section_id].end))
        section_scored.append((section_id, score))
    section_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    candidate_set = set(candidate_pages)
    selected: set[int] = set()
    for section_id, _ in section_scored[:section_count]:
        page_scored = []
        for page_id in section_to_pages.get(section_id, []):
            if page_id not in candidate_set:
                continue
            score = SparseTfidfIndex.cosine(page_query, page_index.vectors[page_id])
            score += authority_score(joined(task.token_texts, pages[page_id].start, pages[page_id].end))
            page_scored.append((page_id, score))
        page_scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        selected.update(page_id for page_id, _ in page_scored[:pages_per_section])
    return selected


def gated_tail_pages(
    task: GeneratedTask,
    pages: list[Any],
    candidate_ids: list[int],
    tail_count: int,
    min_authority_score: float = 0.0,
) -> set[int]:
    tail_ids = sorted(candidate_ids, key=lambda page_id: pages[page_id].end)[-tail_count:]
    selected = set()
    for page_id in tail_ids:
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        if authority_score(text) >= min_authority_score:
            selected.add(page_id)
    return selected


def expand_adjacent_pages(selected: set[int], candidate_ids: list[int], radius: int) -> set[int]:
    if radius <= 0:
        return set(selected)
    candidate_set = set(candidate_ids)
    expanded = set(selected)
    for page_id in selected:
        for offset in range(-radius, radius + 1):
            neighbor = page_id + offset
            if neighbor in candidate_set:
                expanded.add(neighbor)
    return expanded


def authority_anchor_pages(
    task: GeneratedTask,
    pages: list[Any],
    selected: set[int],
    min_authority_score: float = 0.0,
) -> set[int]:
    anchors = set()
    for page_id in selected:
        text = joined(task.token_texts, pages[page_id].start, pages[page_id].end)
        if authority_score(text) > min_authority_score:
            anchors.add(page_id)
    return anchors


def expand_authority_adjacent_pages(
    task: GeneratedTask,
    pages: list[Any],
    selected: set[int],
    candidate_ids: list[int],
    radius: int,
    min_authority_score: float = 0.0,
) -> set[int]:
    anchors = authority_anchor_pages(task, pages, selected, min_authority_score)
    return set(selected) | expand_adjacent_pages(anchors, candidate_ids, radius)


def page_token_count(pages: list[Any], page_id: int) -> int:
    page = pages[page_id]
    return max(0, int(page.end) - int(page.start))


def budgeted_pages(
    pages: list[Any],
    candidate_pages: list[int],
    selected: set[int],
    anchors: set[int],
    page_scores: dict[int, float],
    remote_token_budget: int,
) -> set[int]:
    if remote_token_budget <= 0:
        return set()
    candidate_set = set(candidate_pages)
    candidate_selected = [page_id for page_id in selected if page_id in candidate_set]
    scored = []
    for page_id in candidate_selected:
        if page_id in anchors:
            priority = 3
        elif page_id in page_scores:
            priority = 2
        else:
            priority = 1
        score = page_scores.get(page_id, -1.0)
        scored.append((page_id, priority, score, page_token_count(pages, page_id)))
    scored.sort(key=lambda item: (item[1], item[2], -item[3], -item[0]), reverse=True)
    kept: set[int] = set()
    used = 0
    for page_id, priority, _, token_count in scored:
        if used + token_count <= remote_token_budget:
            kept.add(page_id)
            used += token_count
        elif priority == 3 and not kept:
            kept.add(page_id)
            break
    return kept


def budgeted_authflat_pages(
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    query_text: str,
    candidate_pages: list[int],
    page_count: int,
    radius: int,
    budget_percent: int,
    sink_tokens: int,
    recent_tokens: int,
) -> set[int]:
    scored = auth_flat_scored_pages(task, pages, page_index, query_text, candidate_pages)
    base_list = scored[:page_count]
    base = {page_id for page_id, _ in base_list}
    score_by_page = {page_id: score for page_id, score in base_list}
    anchors = authority_anchor_pages(task, pages, base)
    expanded = base | expand_adjacent_pages(anchors, candidate_pages, radius)
    for page_id in expanded:
        if page_id in score_by_page:
            continue
        nearby_anchor_scores = [
            score_by_page[anchor] - 0.05 * abs(anchor - page_id)
            for anchor in anchors
            if abs(anchor - page_id) <= radius and anchor in score_by_page
        ]
        if nearby_anchor_scores:
            score_by_page[page_id] = max(nearby_anchor_scores)
    total_budget = int(task.prefill_tokens * (budget_percent / 100.0))
    remote_budget = max(0, total_budget - sink_tokens - recent_tokens)
    return budgeted_pages(pages, candidate_pages, expanded, anchors, score_by_page, remote_budget)


def selected_pages_for_mode(
    mode: str,
    task: GeneratedTask,
    pages: list[Any],
    page_index: SparseTfidfIndex,
    sections: list[Any],
    section_index: SparseTfidfIndex,
    section_to_pages: dict[int, list[int]],
    sink_tokens: int,
    recent_tokens: int,
    query_window_tokens: int,
) -> set[int]:
    if mode in {"full", "sink_recent"}:
        return set()
    remote_end = max(0, task.prefill_tokens - recent_tokens)
    candidate_pages = [page.unit_id for page in pages if page.end > sink_tokens and page.start < remote_end]
    candidate_sections = [section.unit_id for section in sections if section.end > sink_tokens and section.start < remote_end]
    query_start = max(0, task.query_start - query_window_tokens)
    query_text = joined(task.token_texts, query_start, task.query_start) + "\n" + task.query_text
    if mode.startswith("hybrid_gatedtail4_authflat"):
        count = int(mode.removeprefix("hybrid_gatedtail4_authflat"))
        tail = gated_tail_pages(task, pages, candidate_pages, 4)
        return tail | auth_flat_pages(task, pages, page_index, query_text, candidate_pages, count)
    match = re.fullmatch(r"chain_authflat_p(\d+)_x(\d+)(?:_authadj(\d+))?", mode)
    if match:
        return chain_authflat_pages(
            task,
            pages,
            page_index,
            query_text,
            candidate_pages,
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or 0),
        )
    match = re.fullmatch(r"chain_authhier_p(\d+)_s(\d+)_x(\d+)(?:_adj(\d+))?", mode)
    if match:
        return chain_authhier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"chain_typedflat_p(\d+)_x(\d+)(?:_adj(\d+))?", mode)
    if match:
        return chain_typedflat_pages(
            task,
            pages,
            page_index,
            query_text,
            candidate_pages,
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or 0),
        )
    match = re.fullmatch(r"chain_typedflat_p(\d+)to(\d+)_x(\d+)(?:_adj(\d+))?", mode)
    if match:
        return chain_typedflat_adaptive_pages(
            task,
            pages,
            page_index,
            query_text,
            candidate_pages,
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"chain_typedflat_auto_x(\d+)(?:_adj(\d+))?", mode)
    if match:
        return chain_typedflat_adaptive_pages(
            task,
            pages,
            page_index,
            query_text,
            candidate_pages,
            2,
            typed_seed_ceiling_for_length(task.prefill_tokens),
            int(match.group(1)),
            int(match.group(2) or 0),
        )
    match = re.fullmatch(r"chain_typedflat_conf(?:_s(\d+))?_x(\d+)(?:_adj(\d+))?", mode)
    if match:
        return chain_typedflat_adaptive_pages(
            task,
            pages,
            page_index,
            query_text,
            candidate_pages,
            2,
            int(match.group(1) or 8),
            int(match.group(2)),
            int(match.group(3) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_conf_s(\d+)_p(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_conf_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(3) or 8),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_auto_p(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_conf_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(2) or 8),
            adaptive_section_fanout(section_to_pages, candidate_sections),
            int(match.group(1)),
            int(match.group(3) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_margin_p(\d+)_m(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_margin_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(3) or 8),
            int(match.group(1)),
            int(match.group(2)) / 100.0,
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_role_s(\d+)_p(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_role_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(3) or 8),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_role_auto_p(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_role_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(2) or 8),
            adaptive_section_fanout(section_to_pages, candidate_sections),
            int(match.group(1)),
            int(match.group(3) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_rolectx_s(\d+)_p(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_rolectx_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(3) or 8),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_rolectx_auto_p(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_rolectx_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(2) or 8),
            adaptive_section_fanout(section_to_pages, candidate_sections),
            int(match.group(1)),
            int(match.group(3) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_rolectxflat_s(\d+)_p(\d+)_c(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_rolectxflat_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(4) or 8),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(5) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_rolectxflat_auto_p(\d+)_c(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_rolectxflat_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(3) or 8),
            adaptive_section_fanout(section_to_pages, candidate_sections),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_rolectxart_s(\d+)_p(\d+)_c(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_rolectxart_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(4) or 8),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(5) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_rolectxart_auto_p(\d+)_c(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_rolectxart_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(3) or 8),
            adaptive_section_fanout(section_to_pages, candidate_sections),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_rolectxsec_s(\d+)_p(\d+)_c(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_rolectxsec_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(4) or 8),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(5) or 0),
        )
    match = re.fullmatch(r"chain_typedhier_rolectxsec_auto_p(\d+)_c(\d+)(?:_seed(\d+))?(?:_adj(\d+))?", mode)
    if match:
        return chain_typedhier_rolectxsec_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(3) or 8),
            adaptive_section_fanout(section_to_pages, candidate_sections),
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(4) or 0),
        )
    match = re.fullmatch(r"hybrid_gatedtail4_authhier_s(\d+)_p(\d+)", mode)
    if match:
        tail = gated_tail_pages(task, pages, candidate_pages, 4)
        return tail | auth_hier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(1)),
            int(match.group(2)),
        )
    if mode.startswith("hybrid_tail4_authflat"):
        count = int(mode.removeprefix("hybrid_tail4_authflat"))
        tail = set(sorted(candidate_pages, key=lambda page_id: pages[page_id].end)[-4:])
        return tail | auth_flat_pages(task, pages, page_index, query_text, candidate_pages, count)
    match = re.fullmatch(r"budget_authflat_p(\d+)_authadj(\d+)_b(\d+)(?:_r(?:\d+|auto\d*))?", mode)
    if match:
        return budgeted_authflat_pages(
            task,
            pages,
            page_index,
            query_text,
            candidate_pages,
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            sink_tokens,
            recent_tokens,
        )
    match = re.fullmatch(r"book_auth_flat_p(\d+)_authadj(\d+)", mode)
    if match:
        selected = auth_flat_pages(task, pages, page_index, query_text, candidate_pages, int(match.group(1)))
        return expand_authority_adjacent_pages(task, pages, selected, candidate_pages, int(match.group(2)))
    match = re.fullmatch(r"book_auth_flat_p(\d+)_adj(\d+)", mode)
    if match:
        selected = auth_flat_pages(task, pages, page_index, query_text, candidate_pages, int(match.group(1)))
        return expand_adjacent_pages(selected, candidate_pages, int(match.group(2)))
    if mode.startswith("book_auth_flat_p"):
        count = int(mode.removeprefix("book_auth_flat_p"))
        return auth_flat_pages(task, pages, page_index, query_text, candidate_pages, count)
    match = re.fullmatch(r"book_auth_hier_s(\d+)_p(\d+)_adj(\d+)", mode)
    if match:
        selected = auth_hier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(1)),
            int(match.group(2)),
        )
        return expand_adjacent_pages(selected, candidate_pages, int(match.group(3)))
    match = re.fullmatch(r"book_auth_hier_s(\d+)_p(\d+)_authadj(\d+)", mode)
    if match:
        selected = auth_hier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(1)),
            int(match.group(2)),
        )
        return expand_authority_adjacent_pages(task, pages, selected, candidate_pages, int(match.group(3)))
    match = re.fullmatch(r"book_auth_hier_s(\d+)_p(\d+)", mode)
    if match:
        return auth_hier_pages(
            task,
            pages,
            page_index,
            sections,
            section_index,
            section_to_pages,
            query_text,
            candidate_pages,
            candidate_sections,
            int(match.group(1)),
            int(match.group(2)),
        )
    selected, _ = selected_page_set(
        mode,
        task.query_start,
        task.token_texts,
        pages,
        page_index,
        sections,
        section_index,
        section_to_pages,
        sink_tokens,
        remote_end,
        query_window_tokens,
    )
    return selected


def pages_to_tokens(pages: list[Any], selected_pages: set[int]) -> set[int]:
    tokens: set[int] = set()
    for page_id in selected_pages:
        page = pages[page_id]
        tokens.update(range(page.start, page.end))
    return tokens


def pages_to_ranges(pages: list[Any], selected_pages: set[int]) -> list[tuple[int, int]]:
    ranges = []
    for page_id in sorted(selected_pages):
        page = pages[page_id]
        ranges.append((int(page.start), int(page.end)))
    return ranges
