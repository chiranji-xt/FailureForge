"""Create a tiny synthetic YOLO dataset (data.yaml + test split) for smoke tests."""
import argparse

from failureforge.demo import make_demo_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    yaml_path = make_demo_dataset(args.out, args.n)
    print(f"wrote {args.n} images; dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
