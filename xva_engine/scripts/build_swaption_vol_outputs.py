from xva_engine.market_data.parsers.swaption_volatility_parser import SwaptionVolatilityParser

def main():
    parser = SwaptionVolatilityParser()

    rows = parser.parse("data/market_data/ir/swaption_volatility_example.csv")

    long_df = parser.to_long_df(rows)
    wide_df = parser.to_wide_df(rows)

    long_df.to_csv("swaption_vol_long_format.csv", index=False)
    wide_df.to_csv("swaption_vol_wide_format.csv", index=False, header=False)

    # Optional Parquet (requires pyarrow or fastparquet)
    try:
        long_df.to_parquet("swaption_vol_long_format.parquet", index=False)
    except ImportError:
        pass

if __name__ == "__main__":
    main()
