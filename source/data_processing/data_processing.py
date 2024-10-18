import json
from typing import Any, List

import pandas as pd


class DataProcessor:

    def __init__(self) -> None:
        self.data_dir = "data/"
        self.project_data_fn = "project_data.csv"
        self.event_data_fn = "data_cleaned.json"
        self.date_cn = "Date"
        self.target_person = "Aleksandar CEBZAN - 9D10341573"

        self.subject_cn = "Subject"
        self.body_cn = "Body"

        self.body_cols = [
            "Project Definition",
            "Comment",
        ]
        self.title_cols = ["Project Description", "Activity Description"]
        self.used_cols = self.body_cols + self.title_cols

        self.event_used_cols = ["id", "subject", "body_clean"]

    def read_data(self, file_name: str) -> pd.DataFrame:
        return pd.read_csv(self.data_dir + file_name, sep=";")

    def process_project_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter data on the target person
        df = df[df["Resource Name - GEID"] == self.target_person]
        # Parse Date column to datetime
        df[self.date_cn] = pd.to_datetime(df[self.date_cn], format="%d.%m.%Y")
        # Split Resource Name - GEID column into Resource Name and Resource ID
        df[["Resource Name", "Resource ID"]] = df[
            "Resource Name - GEID"
        ].str.split(" - ", expand=True)
        # Parse columns to numeric
        df[["posted_time_sale", "posted_time_cost"]] = (
            df[["Posted time (CZK) Sales Rate", "Posted Time (CZK) Cost Rate"]]
            .apply(lambda x: x.str.replace(" ", ""))
            .astype(int)
        )
        # Drop columns
        df.drop(
            columns=[
                "Resource Name - GEID",
                "LDIV",
                "Posted Time (CZK) Cost Rate",
                "Posted time (CZK) Sales Rate",
            ],
            inplace=True,
        )
        return df

    def trim_project_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_trimmed = df[self.used_cols]
        return df_trimmed.drop_duplicates()

    # def build_structure(self, df: pd.DataFrame) -> pd.DataFrame:
    #     df.fillna("", inplace=True)
    #     title = df[self.title_cols].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    #     body = df[self.body_cols].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    #     # return pd.DataFrame({"title": title, "body": body})
    #     print(title)

    def get_project_data(self) -> pd.DataFrame:
        df_proj = self.read_data(file_name=self.project_data_fn)
        df_proj_processed = self.process_project_data(df_proj)
        df_proj_trimmed = self.trim_project_data(df_proj_processed)
        # title_body_df = self.build_structure(df_proj_trimmed)
        return df_proj_trimmed

    def load_event_data(self) -> pd.DataFrame:
        # load from json to df
        with open(f"{self.data_dir}{self.event_data_fn}") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df

    def get_email_data(self) -> Any:
        df = self.load_event_data()
        df_restricted = df[self.event_used_cols]
        data_list = df_restricted.values.tolist()
        return data_list[0:2]


if __name__ == "__main__":
    dp = DataProcessor()
    data = dp.get_email_data()
