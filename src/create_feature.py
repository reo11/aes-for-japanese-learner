from collections import OrderedDict
import re
import pandas as pd
import numpy as np
import MeCab

class PreProcess():
    def __init__(self):
        self.tagger = MeCab.Tagger()
        self.hiragana_pattern = re.compile(r'[あ-ん]+')
        self.katakana_pattern = re.compile(r'[\u30A1-\u30F4]+')
        self.kanji_pattern = re.compile(r'[\u4E00-\u9FD0]+')
        self.column_names = []

    def morph_list(self, text):
        node = self.tagger.parseToNode(text)
        morph = []
        while node is not None:
            k = node.surface
            morph.append(k)
            node = node.next
        return morph

    def pos_list(self, text):
        node = self.tagger.parseToNode(text)
        pos = []
        while node is not None:
            h = node.feature.split(',')[0]
            pos.append(h)
            node = node.next
        return pos
    
    def sub_pos_list(self, text):
        node = self.tagger.parseToNode(text)
        pos = []
        while node is not None:
            h = node.feature.split(',')[1]
            pos.append(h)
            node = node.next
        return pos

    def create_features(self, prompt, text):
        features = OrderedDict()
        # 形態素の抽出
        morph = self.morph_list(text)
        # 品詞の抽出
        pos = self.pos_list(text)
        sub_pos = self.sub_pos_list(text)

        # 総文字数
        features["char_size"] = len(re.sub("[\n\r]", "", text))
        # 総段落数
        features["paragraph_size"] = len(text.splitlines())
        # 総文数
        features["sentence_size"] = len(text.split("。"))
        # 総形態素数
        features["count_morph"] = len(morph)
        # 異なり形態素数
        features["morph_set_size"] = len(set(morph))

        # 第1段落の文字数/全体
        features["first_par_char_rate"] = len(re.sub("[\n\r]", "", text.splitlines()[0])) / features["char_size"]
        # 第1段落の文数/全体
        features["first_par_sentence_rate"] = len(text.splitlines()[0].split("。")) / features["sentence_size"]
        # 第1段落の形態素数/全体
        features["first_par_morph_rate"] = len(self.morph_list(text.splitlines()[0])) / features["count_morph"]
        # 最終段落の文字数/全体
        features["last_par_char_rate"] = len(re.sub("[\n\r]", "", text.splitlines()[-1])) / features["char_size"]
        # 最終段落の文数/全体
        features["last_par_sentence_rate"] = len(text.splitlines()[-1].split("。")) / features["sentence_size"]
        # 最終段落の形態素数/全体
        features["last_par_morph_rate"] = len(self.morph_list(text.splitlines()[-1])) / features["count_morph"]

        # 読点数
        # features["count_comma"] = text.count("、")

        # 段落ごとの
        # 平均読点数
        features["ave_par_comma"] = np.mean([x.count("、") for x in text.splitlines()])
        # 平均文字数
        features["ave_par_char"] = np.mean([len(x) for x in text.splitlines()])
        # 平均形態素数
        features["ave_par_morph"] = np.mean([len(self.morph_list(x)) for x in text.splitlines()])

        # 文ごとの
        # 平均読点数
        # features["ave_sent_comma"] = np.mean([x.count("、") for x in text.splitlines()])
        # # 平均文字数
        # features["ave_sent_char"] = np.mean([len(x) for x in text.splitlines()])
        # # 平均形態素数
        # features["ave_sent_morph"] = np.mean([len(self.morph_list(x)) for x in text.splitlines()])
        
        # 名詞
        features["noun"] = pos.count("名詞") / len(pos)
        # 代名詞
        features["pron"] = pos.count("代名詞") / len(pos)
        # 動詞
        features["verb"] = pos.count("動詞") / len(pos)
        # 副詞
        features["adv"] = pos.count("副詞") / len(pos)
        # 連体詞
        features["conj"] = pos.count("連体詞") / len(pos)
        # 助詞
        features["ppp"] = pos.count("助詞") / len(pos)
        # 数が少ないので一括に
        # # 終助詞(ね)
        # features["eppp_ne"] = sum([m == "ね" and "終助詞" in p for m, p in zip(morph, sub_pos)]) / len(pos)
        # # 終助詞(よ)
        # features["eppp_yo"] = sum([m == "よ" and "終助詞" in p for m, p in zip(morph, sub_pos)]) / len(pos)
        # # 終助詞(の)
        # features["eppp_no"] = sum([m == "の" and "終助詞" in p for m, p in zip(morph, sub_pos)]) / len(pos)
        # # 終助詞(な)
        # features["eppp_na"] = sum([m == "な" and "終助詞" in p for m, p in zip(morph, sub_pos)]) / len(pos)
        # # 終助詞
        features["eppp"] = sum(["終助詞" in p for p in zip(sub_pos)]) / len(pos)
        # 感動詞
        features["interj"] = pos.count("感動詞") / len(pos)
        # 形容詞
        features["adjective"] = pos.count("形容詞") / len(pos)
        # 助動詞
        features["aux"] = pos.count("助動詞") / len(pos)
        # 接続詞
        features["cconj"] = pos.count("接続詞") / len(pos)
        # ひらがな
        features["hiragana"] = sum([self.hiragana_pattern.fullmatch(x) != None for x in text]) / len(text)
        # カタカナ
        features["katakana"] = sum([self.katakana_pattern.fullmatch(x) != None for x in text]) / len(text)
        # 漢字
        features["kanji"] = sum([self.kanji_pattern.fullmatch(x) != None for x in text]) / len(text)

        # プロンプト（お題）と本文のオーバーラップ
        lap = 0
        prompt_morph = self.morph_list(prompt)
        for m in morph:
            if m in prompt_morph:
                lap += 1
        features["prompt_lap"] = lap

        self.column_names = features.keys()
        return features

    def process_multi_data(self, rows):
        multi_features = []
        for prompt, text in rows:
            multi_features.append(self.create_features(prompt, text))
        feature_df = pd.DataFrame(multi_features, columns=self.column_names)
        return feature_df

    def process_df(self, df):
        # df = pd.concat([df, self.process_multi_data(df.loc[:, "text"])], axis=1)
        return self.process_multi_data(df[["prompt", "text"]].values)