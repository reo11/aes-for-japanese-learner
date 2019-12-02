from src.lstm import LSTM
from src.attention import Attention
from src.regressor import AttnRegressor
from src.make_data import DataGenerator
from src.optimize import OptimizedRounder
import pandas as pd
import numpy as np
import warnings
import os
import argparse
import joblib
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='LSTM Model')
parser.add_argument('input_csv', type=str, help="input file must contain 'text_id', 'prompt' and 'text' column")
args = parser.parse_args()

model_type = "LSTM"

def main():
    # gpu or cpuでモデルを定義
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Is cuda available: {torch.cuda.is_available()}")

    test_path = args.input_csv
    test_df = pd.read_csv(test_path)
    result_df = pd.DataFrame()
    result_df["text_id"] = test_df["text_id"]

    cols = ["holistic", "content", "organization", "language"]
    for col in cols:
        # データの読み込み
        Data_gen = DataGenerator(f"./trained_models/LSTM/{col}/word2index")
        lstm = joblib.load(f"./trained_models/LSTM/{col}/lstm")
        regressor = joblib.load(f"./trained_models/LSTM/{col}/regressor")

        # predict test data
        test_path = args.input_csv
        test_df = pd.read_csv(test_path)
        test_df["inputs"] = test_df["prompt"].apply(lambda x: "「" + x + "」") + test_df["text"]
        X_test, y_test = Data_gen.transform(test_df["inputs"].values, np.zeros(len(test_df)))
        test = TensorDataset(torch.Tensor(X_test).to(device), torch.Tensor(y_test).to(device))
        test_loader = DataLoader(test, batch_size=32, shuffle=False)

        pred = []
        attention = []
        test_word_indexes = []
        for x, y in test_loader:
            lstm_outputs = lstm(x)
            output, attn = regressor(lstm_outputs)
            output = output.to("cpu").detach().numpy().flatten()
            a = attn
            pred.extend(output)
            attention.extend(a.to("cpu").detach().numpy())
            test_word_indexes.extend(x.to("cpu").numpy())

        opt_path = f"./trained_models/LSTM/{col}/opt_coef.pkl"
        with open(opt_path, 'rb') as opt_model:
            optR = OptimizedRounder()
            coef = pickle.load(opt_model)
            int_preds = optR.predict(pred, coef)
        result_df[col] = int_preds
    
    result_df.to_csv(f"./output/LSTM.csv", index=False)
    print(result_df)


if __name__ == "__main__":
    main()
