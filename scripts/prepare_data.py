import pandas as pd
from pathlib import Path

DEFAULT_DIR = Path(__file__).parent / "../data"


def prepare_annotation(classes):
    df = pd.read_csv(DEFAULT_DIR / "input" / "target.csv", index_col=0)
    for class_ in classes:
        df_class_annotation = df.copy()
        df_class_annotation['target'] = (df_class_annotation['target'] == class_).astype(int)
        df_class_annotation.to_csv(DEFAULT_DIR / "output/prepare_data" / f"{class_}_target.csv")

if __name__ == "__main__":
    classes = [0,1,2,3]
    prepare_annotation(classes)
