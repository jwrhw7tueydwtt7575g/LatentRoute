import json
import argparse
from datasets import load_dataset


def main(output_file: str = "wiki_corpus.jsonl", max_articles: int = 100_000) -> None:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds["train"]):
            if i >= max_articles:
                break
            record = {"text": sample["text"], "title": sample["title"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i % 1000 == 0:
                print(f"Saved {i} articles...")

    print(f"Done! Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Wikipedia dataset for training.")
    parser.add_argument("--output", type=str, default="wiki_corpus.jsonl", help="Output JSONL file path")
    parser.add_argument("--num_articles", type=int, default=100_000, help="Number of articles to process")

    args = parser.parse_args()
    main(args.output, args.num_articles)
