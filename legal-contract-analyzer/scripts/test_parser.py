"""
test_parser.py — Smoke test for DocumentParserAgent.

Usage:
    python scripts/test_parser.py data/contract_exemplu.pdf [data/contract2.pdf]

Prints:
  - Number of sections detected
  - Number of clauses extracted
  - First 3 ClauseDTO objects (full JSON)
  - Metadata summary
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.parser_agent import DocumentParserAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("test_parser")


def test_contract(pdf_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Testing: {pdf_path}")
    print('='*60)

    agent = DocumentParserAgent()
    dto = agent.parse(pdf_path)

    print(f"\n📄 Metadata:")
    print(f"  Title      : {dto.metadata.title}")
    print(f"  Pages      : {dto.metadata.page_count}")
    print(f"  Parties    : {[p.name for p in dto.metadata.parties]}")
    print(f"  Signing    : {dto.metadata.signing_date}")
    print(f"  Value      : {dto.metadata.value}")
    print(f"  Duration   : {dto.metadata.duration}")

    print(f"\n📑 Structure:")
    print(f"  Sections   : {len(dto.sections)}")
    print(f"  Clauses    : {len(dto.clauses)}")

    if dto.sections:
        print(f"\n  First 3 sections:")
        for s in dto.sections[:3]:
            print(f"    - {s.title} (page {s.start_page})")

    print(f"\n📋 First 3 Clauses (JSON):")
    for clause in dto.clauses[:3]:
        print(json.dumps(clause.model_dump(), ensure_ascii=False, indent=2))

    # Save parsed output
    out_path = f"data/{Path(pdf_path).stem}_parsed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dto.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved to {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_parser.py <path/to/contract.pdf> [...]")
        print("No PDF provided — skipping test.")
        return

    for pdf_path in sys.argv[1:]:
        test_contract(pdf_path)


if __name__ == "__main__":
    main()
