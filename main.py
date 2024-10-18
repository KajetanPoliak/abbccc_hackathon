import json
from pathlib import Path
from typing import Any, Dict, List

from source.pipeline import Pipeline

__here__ = Path(__file__).resolve().parent
__data_dir__ = __here__ / "data"


def run() -> List[Dict[str, Any]]:
    pl = Pipeline()
    results = pl.run()
    return results


if __name__ == "__main__":
    res = run()

    # Save the results to a file
    with open(__data_dir__ / "data_results.json", "w") as file:
        res_json = json.dumps(res, indent=4)
        file.write(res_json)
