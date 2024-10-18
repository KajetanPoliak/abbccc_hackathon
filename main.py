import pandas as pd

from source.pipeline import Pipeline


def run() -> pd.DataFrame:
    pl = Pipeline()
    results = pl.run()
    return results


if __name__ == "__main__":
    res = run()
