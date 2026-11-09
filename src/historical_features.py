
import numpy as np
import pandas as pd


def calculate_historical_features(df, district, season, season_year):
    """
    Calculate historical features for a target season.

    Uses only observations before the target season.
    """

    season_order = {"Maha": 0, "Yala": 1}

    if season not in season_order:
        raise ValueError("Invalid cultivation season")

    # Target season index
    target_index = (
        int(season_year) * 2 + season_order[season]
    )

    # Filter selected district
    history = df[df["district"] == district].copy()

    # Create chronological season index
    history["season_index"] = (
        history["season_year"] * 2
        + history["season"].map(season_order)
    )

    # Exclude current and future seasons
    history = history[
        history["season_index"] < target_index
    ]

    history = history.sort_values("season_index")

    if history.empty:
        return {
            "season_gap": np.nan,
            "previous_yield": np.nan,
            "yield_3season_avg": np.nan
        }

    last_index = int(history["season_index"].iloc[-1])

    season_gap = target_index - last_index

    # Map each historical season to its yield
    yield_by_season = history.set_index(
        "season_index"
    )["average_yield_kg_ha"]

    # Immediately preceding season
    previous_yield = yield_by_season.get(
        target_index - 1,
        np.nan
    )

    # Three consecutive preceding seasons
    previous_three = [
        yield_by_season.get(target_index - i, np.nan)
        for i in range(1, 4)
    ]

    if all(pd.notna(value) for value in previous_three):
        avg_three = float(np.mean(previous_three))
    else:
        avg_three = np.nan

    return {
        "season_gap": float(season_gap),
        "previous_yield": float(previous_yield)
            if pd.notna(previous_yield) else np.nan,
        "yield_3season_avg": avg_three
    }
