import pandas as pd

SEED = 42


def get_data():
    data_df = pd.read_csv("all_data.csv", index_col=0)
    return data_df


def collect_dataset():
    data_df = get_data()

    num_per_group = 6  # it has 10 semantic categories

    sampled_df = data_df.groupby("Category", group_keys=False).apply(
        lambda group: group.sample(n=num_per_group, random_state=SEED)
    )
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df.rename(columns={"Goal": "task"}, inplace=True)

    sampled_df.to_csv("data.csv")


if __name__ == "__main__":
    collect_dataset()
