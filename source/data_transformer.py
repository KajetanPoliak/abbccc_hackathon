import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from tqdm.auto import tqdm

__here__ = Path(__file__).resolve().parent
__root__ = __here__.parents[0]
__data_dir__ = __root__ / "data"
assert __data_dir__.exists(), f"Data directory not found: {__data_dir__!r}"


class CalendarMeeting:

    datetime_format = "%Y-%m-%dT%H:%M:%S%z"
    date_format = "%Y-%m-%d"

    def __init__(self, raw: Dict[str, Any]) -> None:
        self.raw = raw
        self.id = raw["id"]
        self.ical_uid = raw["iCalUId"]
        self.subject = raw["subject"]
        self.body_preview = raw["bodyPreview"]
        self.body = raw["body"]["content"]
        self.body_clean = self.clean_body()
        self.body_preview_clean = self.clean_body_preview()
        self.start: datetime.datetime = datetime.datetime.strptime(
            raw["start"]["dateTime"], "%Y-%m-%dT%H:%M:%S.%f0"
        )
        self.end: datetime.datetime = datetime.datetime.strptime(
            raw["end"]["dateTime"], "%Y-%m-%dT%H:%M:%S.%f0"
        )
        self.timezone = raw["start"]["timeZone"]
        self.duration_hours = (self.end - self.start).total_seconds() / 3600
        self.instances: List[datetime.date] = []
        self.process_recurrence()

    def process_recurrence(self) -> None:
        if self.raw["recurrence"] is None:
            self.instances = [(self.start.date())]
        else:
            rec = self.raw["recurrence"]
            if rec["pattern"]["type"] != "weekly":
                raise Exception("Monthly recurrence -> Out of Scope... :D")
            # replace weekdays with corresponding next dates
            occ = rec["pattern"]["daysOfWeek"].copy()
            # iterate over the array and granually increment the dates and fill self.instances
            for i in range(0, 7):  # all possible weekdays
                curr = self.start + datetime.timedelta(days=i)
                if (self.start + datetime.timedelta(days=i)).strftime(
                    "%A"
                ).lower() in occ:
                    occ[occ.index(curr.strftime("%A").lower())] = (
                        self.start + datetime.timedelta(days=i)
                    )
            while (
                True
            ):  # append to instances the minimal day in occ, after appending increase the date by interval * 7 days (all recurrences are weekly)
                min_date = min(occ)
                # check end condition
                if (
                    datetime.datetime.strptime(
                        rec["range"]["endDate"], "%Y-%m-%d"
                    ).date()
                    < min_date.date()
                ):
                    break
                self.instances.append(min_date.date())
                min_date_idx = occ.index(min_date)
                occ[min_date_idx] = (
                    min_date
                    + datetime.timedelta(days=7) * rec["pattern"]["interval"]
                )

    def clean_body(self) -> str:
        text = BeautifulSoup(markup=self.body, features="html.parser").text
        text = re.sub(r"(\r\n|\n|\r){4,}", r"\n", text)
        text = re.sub(r"_{10,}.*_{10,}", " ", text, flags=re.DOTALL)
        text = re.sub(r"-{3,}", " ", text, flags=re.DOTALL)
        text = re.sub(r"[\n\r]", " ", text, flags=re.DOTALL)
        text = re.sub(r"\s{2,}", " ", text, flags=re.DOTALL)
        if not text:
            return ""
        return text.strip("\n ")

    def clean_body_preview(self) -> str:
        text = re.sub(r"_{10,}.*", " ", self.body_preview, flags=re.DOTALL)
        text = re.sub(r"-{3,}", " ", text, flags=re.DOTALL)
        text = re.sub(r"(\r\n|\n|\r){4,}", r"\n", text)
        text = re.sub(r"[\n\r]", " ", text, flags=re.DOTALL)
        text = re.sub(r"\s{2,}", " ", text, flags=re.DOTALL)
        if not text:
            return ""
        return text.strip("\n ")

    def __repr__(self) -> str:
        return self.to_json()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ical_uid": self.ical_uid,
            "subject": self.subject,
            "body_preview_clean": self.body_preview_clean,
            "body_clean": self.body_clean,
            "start": self.start.strftime(self.datetime_format),
            "end": self.end.strftime(self.datetime_format),
            "duration": float(self.duration_hours),
            "instances": [
                date.strftime(self.date_format) for date in self.instances
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


if __name__ == "__main__":
    with open(__data_dir__ / "data.json", encoding="utf-8") as f:
        data = json.load(f)

    outputs = []
    for i in tqdm(range(0, len(data["value"]))):
        x = CalendarMeeting(raw=data["value"][i])
        outputs.append(x.to_dict())

    # Save to JSON
    with open(__data_dir__ / "data_cleaned.json", "w") as f:
        f.write(json.dumps(outputs, indent=4))
