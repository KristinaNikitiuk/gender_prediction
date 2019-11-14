from age_gender.prepare_data import PrepareData
from age_gender.random_forest import RandomForest


def main():
    prepare_data = PrepareData()
    RandomForest(prepare_data.normalize_data()).run_model()


if __name__ == "__main__":
    main()