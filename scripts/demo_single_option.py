from xva_engine.orchestration.workflows import run_single_equity_option_demo


def main():
    params = {
        "spot": 100.0,
        "rate": 0.02,
        "sigma": 0.20,
        "maturity": 1.0,
        "strike": 100.0,
        "n_scenarios": 10000,
        "n_steps": 50,
        "t_max": 1.0,
    }

    result = run_single_equity_option_demo(params)

    print("EE(t) at a few points:")
    times = result.rf_cube.time_grid.times
    for idx in [0, 10, 25, 50]:
        print(f"t={times[idx]:.2f}, EE={result.ee[idx, 0]:.4f}")

    print("\nPFE(95%) at maturity:")
    print(result.pfe_95[-1, 0])


if __name__ == "__main__":
    main()
