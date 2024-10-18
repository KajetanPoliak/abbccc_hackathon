import pandas as pd
import json
import datetime

class CalendarMeeting:
    
    def __init__(self, raw):
        self.raw = raw
        self.subject = raw["subject"]
        self.body_preview = raw["bodyPreview"]
        self.body = raw["body"]
        self.start = datetime.datetime.strptime(raw["start"]["dateTime"], "%Y-%m-%dT%H:%M:%S.%f0")
        self.end = datetime.datetime.strptime(raw["end"]["dateTime"], "%Y-%m-%dT%H:%M:%S.%f0")
        self.timezone = raw["start"]["timeZone"]
        self.duration_hours = (self.end - self.start).seconds / 60 / 60
        self.instances = []
        self.process_recurrence()

    def process_recurrence(self):
        if self.raw["recurrence"] is None:
            self.instances = [(self.start.date())]
        else:
            rec = self.raw["recurrence"]
            if rec["pattern"]["type"] != "weekly":
                raise Exception("Monthly recurrence -> Out of Scope... :D")
            # replace weekdays with corresponding next dates
            occ = rec["pattern"]["daysOfWeek"].copy()
            # iterate over the array and granually increment the dates and fill self.instances
            for i in range(0, 7): #all possible weekdays
                curr = self.start + datetime.timedelta(days=i)
                if (self.start + datetime.timedelta(days=i)).strftime('%A').lower() in occ:
                    occ[occ.index(curr.strftime("%A").lower())] = self.start + datetime.timedelta(days=i)
            while True: # append to instances the minimal day in occ, after appending increase the date by interval * 7 days (all recurrences are weekly)
                min_date = min(occ)
                # check end condition
                if datetime.datetime.strptime(rec["range"]["endDate"], "%Y-%m-%d").date() < min_date.date():
                    break
                self.instances.append(min_date.date())
                min_date_idx = occ.index(min_date)
                occ[min_date_idx] = min_date + datetime.timedelta(days=7) * rec["pattern"]["interval"]          

    def clean_body_preview(self):
        pass

    def clean_body(self):
        pass

    def __repr__(self):
        return f"""Start: {self.start}
End: {self.end}
Duration: {float(self.duration_hours)}
Instances: {self.instances}"""

with open("./data/data.json", encoding="utf-8") as f:
    data = json.load(f)

for i in range(0, len(data["value"])):
    print(i, "################################")
    x = CalendarMeeting(raw=data["value"][i])
    print(x)
