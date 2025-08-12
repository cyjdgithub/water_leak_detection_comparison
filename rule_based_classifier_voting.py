# rule_based_classifier.py

import ast
import numpy as np
from collections import Counter

class RuleBasedClassifier:
    def __init__(self):
        self.rules = []

    def fit(self, rules):
        """
        支持多个目标类的规则，例如同时包含 leak_label__1.0 和 leak_label__0.0 的规则。
        """
        self.rules = rules if isinstance(rules, list) else rules.to_dict("records")

    def predict(self, X):
        """
        Predict using rule-based voting.
        X: pandas DataFrame（必须包含所有规则使用到的特征）
        返回：预测标签（float），比如 1.0 或 0.0
        """
        results = []

        for _, row in X.iterrows():
            matched_votes = []

            for rule in self.rules:
                if self._match_antecedents(rule["antecedents"], row):
                    label = rule["consequent"].split("__")[1]
                    confidence = float(rule.get("confidence", 1.0))
                    # Append (label, confidence)
                    matched_votes.append((label, confidence))

            if len(matched_votes) == 0:
                results.append(0.0)  # 默认无命中时预测为 0
            else:
                # Voting with confidence weighting
                vote_counter = {}
                for label, conf in matched_votes:
                    vote_counter[label] = vote_counter.get(label, 0.0) + conf

                # Select label with highest total confidence
                voted_label = max(vote_counter.items(), key=lambda x: x[1])[0]
                results.append(float(voted_label))

        return results

    def _match_antecedents(self, antecedents, row):
        if isinstance(antecedents, str):
            antecedents = ast.literal_eval(antecedents)

        for condition in antecedents:
            try:
                feature, value = condition.split("__")
                if str(row[feature]) != value:
                    return False
            except Exception:
                return False
        return True
