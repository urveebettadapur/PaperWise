import json
import sys
from pathlib import Path


def main():

    file_path = Path("evaluation_candidates.json")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    question_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    if question_number < 1 or question_number > len(data):
        print("Invalid question number.")
        return

    item = data[question_number - 1]

    print(f"\nQUESTION {question_number}")
    print("=" * 80)
    print(item["question"])

    for index, candidate in enumerate(item["candidates"], start=1):

        print("\n" + "-" * 80)
        print(f"CANDIDATE {index}")
        print(f"Chunk ID: {candidate['chunk_id']}")
        print(f"Retrieved by: {', '.join(candidate['retrieved_by'])}")
        print("\nTEXT:")
        print(candidate["text"])


if __name__ == "__main__":
    main()