"""
NeuralPath — O*NET Knowledge Graph Seeder
==========================================
Seeds the NeuralPath knowledge graph with O*NET occupation data.

This script does NOT require a database. It generates a JSON snapshot
of the 73-node Knowledge Graph enriched with O*NET SOC codes, so the
data can be inspected, exported, or used by other tools.

Usage:
  python scripts/seed_onet.py                    # print JSON to stdout
  python scripts/seed_onet.py --out graph.json   # write to file
  python scripts/seed_onet.py --stats            # print stats only

O*NET Data:
  Source:  https://www.onetcenter.org/db_releases.html
  Version: O*NET 28.3 (2024)
  License: Public Domain (U.S. Government Work)

The O*NET SOC codes embedded in knowledge_graph.py were derived from
the O*NET 28.3 database. This script exposes them for transparency.
"""

import sys
import json
import argparse
from pathlib import Path

# ── Add project root to path so we can import backend ─────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def build_graph_export() -> dict:
    """Build a full JSON export of the Knowledge Graph with O*NET metadata."""
    from backend.knowledge_graph import SKILL_NODES, KNOWLEDGE_GRAPH
    import networkx as nx

    nodes = []
    for skill in SKILL_NODES:
        nodes.append({
            "id":            skill.id,
            "name":          skill.name,
            "description":   skill.description,
            "domain":        skill.domain,
            "difficulty":    skill.difficulty,
            "base_hours":    skill.base_hours,
            "tags":          skill.tags,
            "onet_codes":    skill.onet_codes,
            "prerequisites": skill.prerequisites,
            "in_degree":     KNOWLEDGE_GRAPH.in_degree(skill.id),
            "out_degree":    KNOWLEDGE_GRAPH.out_degree(skill.id),
        })

    edges = [
        {
            "source": u,
            "target": v,
            "weight": data.get("weight", 1),
            "relationship": data.get("relationship", "prerequisite"),
        }
        for u, v, data in KNOWLEDGE_GRAPH.edges(data=True)
    ]

    # Domain breakdown
    domains: dict[str, int] = {}
    for skill in SKILL_NODES:
        domains[skill.domain] = domains.get(skill.domain, 0) + 1

    # O*NET code coverage
    onet_covered = [s for s in SKILL_NODES if s.onet_codes]

    return {
        "metadata": {
            "version":        "2.0.0",
            "onet_version":   "28.3 (2024)",
            "onet_source":    "https://www.onetcenter.org/db_releases.html",
            "onet_license":   "Public Domain (U.S. Government Work)",
            "total_nodes":    len(nodes),
            "total_edges":    len(edges),
            "domains":        domains,
            "onet_coverage":  f"{len(onet_covered)}/{len(SKILL_NODES)} nodes have O*NET codes",
        },
        "nodes": nodes,
        "edges": edges,
    }


def print_stats(export: dict) -> None:
    """Print human-readable stats."""
    m = export["metadata"]
    print(f"\nNeuralPath Knowledge Graph — O*NET Export")
    print(f"{'─' * 50}")
    print(f"Total nodes:      {m['total_nodes']}")
    print(f"Total edges:      {m['total_edges']}")
    print(f"O*NET version:    {m['onet_version']}")
    print(f"O*NET coverage:   {m['onet_coverage']}")
    print(f"\nDomain breakdown:")
    for domain, count in sorted(m["domains"].items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {domain:20} {bar} ({count})")

    print(f"\nSample nodes with O*NET codes:")
    for node in export["nodes"]:
        if node["onet_codes"]:
            print(f"  {node['name']:35} → {', '.join(node['onet_codes'])}")
    print()


def main():
    parser = argparse.ArgumentParser(description="NeuralPath O*NET Knowledge Graph Seeder")
    parser.add_argument("--out",   help="Output JSON file path (default: stdout)")
    parser.add_argument("--stats", action="store_true", help="Print stats only, no JSON output")
    args = parser.parse_args()

    print("Building Knowledge Graph export...", file=sys.stderr)
    export = build_graph_export()

    if args.stats:
        print_stats(export)
        return

    json_str = json.dumps(export, indent=2, ensure_ascii=False)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"✓ Written to {out_path} ({out_path.stat().st_size // 1024}KB)", file=sys.stderr)
        print_stats(export)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
