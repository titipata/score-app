import spacy
from transformers import pipeline


class SCOREPredictor:
    """
    HuggingFace's predictor for SCORE claim dataset.
    Use this API to predict the claim type of sentences in a given paragraph.

    >>> model = SCOREPredict()
    >>> model.load_model()
    >>> results = model.predict(paragraph)
    """

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

    def split_paragraph(self, paragraph: str):
        """Split paragraph into sentences"""
        paragraph = self.NLP(paragraph)
        sents = [sent.text for sent in paragraph.sents]
        return sents

    def predict(self, model_name: str, paragraph: str):
        """Predict the claim type of sentences in a given paragraph."""
        assert (
            model_name in self.CLASSIFIERS.keys()
        ), "A given model is not found in list of classifier."
        assert (
            len(self.CLASSIFIERS) == 0
        ), "You have not downloaded the model yet. Please run `load_model` first."
        classifier = self.CLASSIFIERS.get(model_name, None)
        if classifier is None:
            raise Exception("Invalid model")

        sents = self.split_paragraph(paragraph)
        results = classifier(sents, return_all_scores=True)

        preds = []
        for i, result in enumerate(results):
            sent = sents[i]
            preds.append({"sent": sent, "result": result})
        return preds
