# rule_based_classifier.py

import ast
import numpy as np
from collections import Counter

class RuleBasedClassifier:
    def __init__(self):
        self.rules = []

    def fit(self, rules):

        self.rules = rules if isinstance(rules, list) else rules.to_dict("records")

    def predict(self, X):

        results = []

        for _, row in X.iterrows():
            matched_labels = []

            for rule in self.rules:
                if self._match_antecedents(rule["antecedents"], row):
                    label = rule["consequent"].split("__")[1]
                    matched_labels.append(label)

            if len(matched_labels) == 0:
                results.append(0.0)
            else:
                most_common = Counter(matched_labels).most_common(1)[0][0]
                results.append(float(most_common))

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