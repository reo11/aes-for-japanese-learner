import pickle
import pandas as pd
from src.create_feature import PreProcess
from src.optimize import OptimizedRounder
import argparse

parser = argparse.ArgumentParser(description='Random Forest Model')
parser.add_argument('input_csv', type=str, help="input file must contain 'text_id', 'prompt' and 'text' column")
args = parser.parse_args()

model_name = "random_forest"

if __name__ == '__main__':
    test_path = args.input_csv
    test_df = pd.read_csv(test_path)
    pp = PreProcess()
    X = pp.process_df(test_df).values
    result_df = pd.DataFrame()
    result_df["text_id"] = test_df["text_id"]
    for col in ["holistic", "content", "organization", "language"]:
        clf_path = f"./trained_models/{model_name}/{col}/clf.pkl"
        opt_path = f"./trained_models/{model_name}/{col}/opt_coef.pkl"
        with open(clf_path, 'rb') as clf_model, open(opt_path, 'rb') as opt_model:
            clf = pickle.load(clf_model)
            optR = OptimizedRounder()
            coef = pickle.load(opt_model)
            preds = clf.predict(X)
            int_preds = optR.predict(preds, coef)
            result_df[col] = int_preds
    result_df.to_csv(f"./output/{model_name}.csv", index=False)
    print(result_df)
