from pathlib import Path
from  market_data.parsers.yield_curve_parser import YieldCurveParser

ROOT = Path(__file__).resolve().parents[1]   # project root
DATA = ROOT / "data" / "market_data" / "yield_curves"


def main():
    parser = YieldCurveParser()

    rows = parser.parse(DATA/"yield_curve_example.csv")  # or your dataset file
    long_df = parser.to_long_df(rows)
    wide_df = parser.to_wide_df(rows)

    long_df.to_csv("yield_curves_long_format.csv", index=False)
    wide_df.to_csv("yield_curves_wide_format.csv", index=False, header=False)

    # Safer than pickle across numpy versions:
    long_df.to_parquet("yield_curves_long_format.parquet", index=False)

if __name__ == "__main__":
    main()
