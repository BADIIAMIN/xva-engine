from market_data.parsers.credit_spread_parser import CreditSpreadParser

def main():
    parser = CreditSpreadParser()
    rows = parser.parse("data/market_data/credit/credit_spread_example.csv")

    long_df = parser.to_long_df(rows)
    wide_df = parser.to_wide_df(rows)

    long_df.to_csv("credit_curves_long_format.csv", index=False)
    wide_df.to_csv("credit_curves_wide_format.csv", index=False, header=False)

    # Prefer Parquet, but only if pyarrow installed:
    try:
        long_df.to_parquet("credit_curves_long_format.parquet", index=False)
    except ImportError:
        pass

if __name__ == "__main__":
    main()
