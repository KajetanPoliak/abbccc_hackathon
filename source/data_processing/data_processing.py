import pandas as pd


class DataProcessor:

    def __init__(self) -> None:
        self.data_dir = "data/"
        self.project_data_fn = "project_data.csv"
        self.date_cn = "Date"
        self.target_person = "Aleksandar CEBZAN - 9D10341573"

        self.used_cols = [
            "Project Description",
            "Project Definition",
            "Activity Description",
            "Comment",
        ]

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

    def trim_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_trimmed = df[self.used_cols]
        return df_trimmed.drop_duplicates()


if __name__ == "__main__":
    dp = DataProcessor()
    df = dp.read_data("project_data.csv")
    df_processed = dp.process_project_data(df)
    print(df_processed.shape)

    df_trimmed = dp.trim_data(df_processed)
    print(df_trimmed.shape)
    print(df_trimmed.head())
    df_trimmed.to_csv("data/trimmed_project_data.csv", index=False)
