from datetime import date
from xva_engine.market_data.simulators.yield_curve_dummy_simulator import (
    YieldCurveDummySimulator, DummyCurveSpec
)
from xva_engine.market_data.sources.yield_curve_csv_source import YieldCurveCsvSource

def main():
    # 1) simulate dummy dataset
    sim = YieldCurveDummySimulator(seed=1)
    specs = [
        DummyCurveSpec(curve_id="EUR_OIS", currency="EUR"),
        DummyCurveSpec(curve_id="USD_OIS", currency="USD"),
    ]
    df = sim.generate_dataset(specs, start_date=date(2025, 1, 1), n_days=5)
    sim.to_csv(df, "dummy_yield_curves.csv")

    # 2) load it via MarketDataSource
    src = YieldCurveCsvSource("dummy_yield_curves.csv", has_header=False)

    # 3) snapshot
    env = src.get_snapshot("01/01/2025")
    eur = env.get_curve("EUR_OIS")

    print("EUR_OIS 2Y zero:", eur.zero_rate(730))
    print("EUR_OIS 5Y DF:", eur.df(1825))

if __name__ == "__main__":
    main()
