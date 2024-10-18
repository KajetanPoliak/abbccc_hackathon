import pandas as pd

DATA_DIR = "data/"


def read_data(file_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR + file_name, sep=";")


def process_project_data(df: pd.DataFrame) -> pd.DataFrame:
    # Parse Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    # Split Resource Name - GEID column into Resource Name and Resource ID
    df[["Resource Name", "Resource ID"]] = df["Resource Name - GEID"].str.split(
        " - ", expand=True
    )
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


if __name__ == "__main__":
    df = read_data("project_data.csv")
    df_processed = process_project_data(df)
    print(df_processed.head())
    print(df_processed.info())
