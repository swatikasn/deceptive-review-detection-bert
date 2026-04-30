import argparse
from pathlib import Path

from lie_model import DeceptionModel, demo_rows, load_csv_rows, save_model


DEFAULT_MODEL_PATH = Path("models/deception_model.pkl")


def main():
    parser = argparse.ArgumentParser(description="Train and save the lie detection text model.")
    parser.add_argument(
        "--data",
        help="CSV containing review text and labels. Supports text/text_/review and label/target columns.",
    )
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH), help="Where to save the model.")
    args = parser.parse_args()

    if args.data:
        rows = load_csv_rows(args.data)
        source = Path(args.data).name
    else:
        rows = demo_rows()
        source = "built-in demo seed data"

    model = DeceptionModel().fit(rows, source=source)
    save_model(model, args.output)
    print(f"Saved model to {args.output}")
    print(f"Training rows: {model.trained_rows}")
    print(f"Source: {model.source}")


if __name__ == "__main__":
    main()
