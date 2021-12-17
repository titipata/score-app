from transformers import pipeline

import os
import spacy


class SCOREPredictor:
    def __init__(self) -> None:

        self.NLP = spacy.load("en_core_web_sm")

        self.MODEL_DICT = {
            "claim2": {"path": "SCORE/claim2-distilbert-base-uncased"},
            "claim3a": {"path": "SCORE/claim3a-distilbert-base-uncased"},
            "claim3b": {"path": "SCORE/claim3b-distilbert-base-uncased"},
        }

        self.CLASSIFIERS = {}

    def load_model(self):
        for model_name, item in self.MODEL_DICT.items():

            print(f"Loading model: {model_name}")
            self.CLASSIFIERS[model_name] = pipeline(
                "text-classification", model=item["path"]
            )

    def paragraph_to_sentences(self, paragraph: str):

        paragraph = self.NLP(paragraph)

        sents = [sent.text for sent in paragraph.sents]

        return sents

    def predict(self, model_name: str, paragraph: str):

        classifier = self.CLASSIFIERS.get(model_name, None)

        if classifier is None:
            raise Exception("Invalid model")

        sents = self.paragraph_to_sentences(paragraph)

        results = classifier(sents, return_all_scores=True)

        final_result = []
        for i, result in enumerate(results):
            sent = sents[i]
            final_result.append({"sent": sent, "result": result})

        return final_result
