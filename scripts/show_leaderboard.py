"""Pretty-print the last training leaderboard."""
import json
import sys
from pathlib import Path


def main():
    p = Path(__file__).resolve().parent.parent / "results" / "metrics" / "leaderboard.json"
    if not p.exists():
        print(f"No leaderboard at {p}. Run `make train` first.")
        sys.exit(1)
    data = json.loads(p.read_text())
    print(f"\n{'Model':<30} {'Task':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'F1':>8}")
    print("-" * 90)
    for row in data:
        print(
            f"{row.get('model', '-'):<30} "
            f"{row.get('task', '-'):<25} "
            f"{row.get('rmse', 0):>8.3f} "
            f"{row.get('mae', 0):>8.3f} "
            f"{row.get('r2', 0):>8.3f} "
            f"{row.get('f1_weighted', 0):>8.3f}"
        )
    print()


if __name__ == "__main__":
    main()
