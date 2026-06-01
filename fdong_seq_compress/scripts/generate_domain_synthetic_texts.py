from __future__ import annotations

from pathlib import Path


OUT_DIR = Path("fdong_seq_compress/data/synthetic_texts")


def write_text(name: str, sections: list[str]) -> None:
    path = OUT_DIR / name
    text = "\n\n".join(sections).strip() + "\n"
    path.write_text(text, encoding="utf-8")
    print(f"{path}\twords={len(text.split())}\tchars={len(text)}")


def make_textbook() -> list[str]:
    sections = [
        "Textbook Chapter: Distributed Systems, Memory Hierarchies, and Retrieval.\n"
        "This chapter introduces a sequence of connected concepts that reappear across later examples: process, message, clock, cache, index, consistency, checkpoint, and recovery."
    ]
    topics = [
        ("logical clocks", "event ordering", "message causality", "happens-before relation"),
        ("replication", "leader election", "quorum read", "write conflict"),
        ("caching", "eviction policy", "locality", "stale read"),
        ("database indexing", "B tree", "inverted index", "range query"),
        ("fault tolerance", "checkpoint", "replay log", "idempotent operation"),
        ("distributed tracing", "span id", "root cause", "latency percentile"),
    ]
    for i in range(90):
        a, b, c, d = topics[i % len(topics)]
        sections.append(
            f"Section {i+1}: {a.title()}.\n"
            f"The section defines {a} through a concrete system that receives client requests, updates internal state, and emits observable logs. "
            f"The concept of {b} is introduced first as a local rule and later reused as a global invariant. "
            f"A recurring example called Service-{i%7} processes request R{i:04d}, stores checkpoint C{i%11}, and references index entry I{i%13}. "
            f"The chapter deliberately repeats the names process, message, clock, cache, index, consistency, checkpoint, and recovery so that long-range references can be tested. "
            f"When {c} changes, the interpretation of {d} also changes, but the earlier definition remains necessary. "
            f"Students are asked to compare the current section with Section {max(1, i-5)} and Section {max(1, i-17)}, because the same mechanism appears under a different failure model. "
            f"The mathematical statement is that local transitions compose into a global trace, but the trace is only useful when it can be searched by meaningful anchors. "
            f"Example {i+1} ends by recording lemma L{i%19}, invariant V{i%23}, and counterexample X{i%29}, which are cited again in later exercises."
        )
    return sections


def make_codebase() -> list[str]:
    sections = [
        "Repository Notes: Query Engine, Cache Manager, and Graph Planner.\n"
        "This synthetic codebase document interleaves API descriptions, pseudo-code, bug reports, and design notes."
    ]
    modules = [
        ("CachePage", "pin_page", "evict_cold_page", "page_table"),
        ("GraphIndex", "insert_node", "expand_neighbors", "adjacency_list"),
        ("QueryPlanner", "estimate_cost", "choose_candidate_set", "plan_cache"),
        ("KVReader", "read_key_block", "read_value_block", "block_handle"),
        ("TokenizerBridge", "map_token_span", "decode_piece", "offset_table"),
    ]
    for i in range(100):
        cls, f1, f2, field = modules[i % len(modules)]
        sections.append(
            f"File src/{cls.lower()}_{i%9}.py.\n"
            f"class {cls}{i%5}: the class owns field `{field}_{i%17}` and exposes `{f1}` and `{f2}`. "
            f"The method `{f1}` receives request_id={1000+i}, layer_id={i%28}, head_id={i%8}, and token_span=({i*16}, {i*16+63}). "
            f"The method `{f2}` must not drop exact value payloads when the selected key index is only approximate. "
            f"Pseudo-code: if score_margin < threshold_{i%13}, call fallback_exact_read(block_handle); otherwise gather candidate_values from the selected block list. "
            f"Bug note BUG-{i:04d}: a stale `{field}_{i%17}` entry caused a query to reuse an old graph neighborhood after the cache was compacted. "
            f"Design note: keys are allowed to be compressed for routing, but values remain high-fidelity until the final read path. "
            f"The unit test `test_{cls.lower()}_{f1}_{i%31}` references earlier bug BUG-{max(0, i-12):04d} and later regression REG-{i%37}."
        )
    return sections


def make_news_dossier() -> list[str]:
    sections = [
        "News Dossier: Semiconductor Supply Chain, Energy Demand, and Regional Policy.\n"
        "This dossier contains many article-like entries about overlapping events, people, organizations, dates, and claims."
    ]
    regions = ["Taiwan", "Arizona", "Saxony", "Singapore", "Seoul", "Bangalore"]
    orgs = ["Northbridge Foundry", "Helios Grid", "Mariner Logistics", "Aster Research", "Civic Energy Office"]
    for i in range(96):
        region = regions[i % len(regions)]
        org = orgs[i % len(orgs)]
        sections.append(
            f"Article {i+1}: {org} expands planning office in {region}.\n"
            f"Officials said the decision was influenced by power availability, water permits, skilled labor, and transport reliability. "
            f"The article cites memorandum M-{i%25:02d}, policy docket P-{i%18:02d}, and shipment record S-{i%33:02d}. "
            f"A later paragraph repeats the same organization under a different role: supplier, customer, regulator, investor, or grid operator. "
            f"Analysts compare this story with Article {max(1, i-9)} and Article {max(1, i-21)} because both mention the same bottleneck under a different economic assumption. "
            f"The dossier records uncertainty explicitly: one source confirms the permit, another disputes the construction timeline, and a third notes that demand forecasts changed after a heat wave. "
            f"The important retrieval challenge is that a question about {region} may require a policy fact, a logistics fact, and an energy fact that were introduced many articles apart."
        )
    return sections


def make_dialogue_tools() -> list[str]:
    sections = [
        "Agent Transcript: Long Conversation With Tool Results.\n"
        "The transcript alternates between user requests, assistant reasoning summaries, tool outputs, and follow-up corrections."
    ]
    tasks = ["budget model", "paper summary", "database migration", "travel plan", "incident report", "code review"]
    for i in range(110):
        task = tasks[i % len(tasks)]
        sections.append(
            f"Turn {i+1} USER: Please update the {task} using constraint C{i%14}, previous decision D{i%19}, and evidence item E{i%23}.\n"
            f"Turn {i+1} ASSISTANT: I will preserve the exact constraint and check whether it conflicts with decision D{max(0, i-7)%19}. "
            f"Tool result {i+1}: table rows include item_id={2000+i}, timestamp=2026-06-{(i%28)+1:02d}, status={'open' if i%3 else 'closed'}, owner=team_{i%8}. "
            f"Correction {i+1}: the user clarifies that evidence item E{i%23} should be interpreted as a hard requirement, not as a preference. "
            f"Memory note {i+1}: this turn refers back to Turn {max(1, i-13)} and will be cited again by Turn {i+17}. "
            f"The transcript creates repeated local forms but long-range dependencies over constraints, owners, decisions, and exact identifiers."
        )
    return sections


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_text("long_textbook_distributed_systems.txt", make_textbook())
    write_text("long_codebase_query_engine.txt", make_codebase())
    write_text("long_news_supply_chain_dossier.txt", make_news_dossier())
    write_text("long_dialogue_tool_transcript.txt", make_dialogue_tools())


if __name__ == "__main__":
    main()
