import streamlit as st
from annotated_text import annotated_text

from api_hg import SCOREPredictor
import scipdf
import json
import traceback
import sys
import hashlib

anno_texts = ["Result"]

def display_result(result):

    anno_texts = []
    for r in result:
        sent = r["sent"]

        claim_score = r["result"][1]["score"]

        if claim_score < 0.5:
            anno_texts.append(sent)
            continue

        anno_texts.append((sent, "CLAIM {}%".format(round(claim_score * 100, 2)), "#8ef"))

    print(anno_texts)

    st.markdown("** Result: **")
    # "Result: "
    annotated_text(*anno_texts)

def parse_pdf(pdf_content):
    try:
        parsed_article = scipdf.parse_pdf_to_dict(
            pdf_content, grobid_url="https://cloud.science-miner.com/grobid/"
        )
        title = parsed_article.get("title", "")
        contents = []
        contents.append(
            {
                "id": 0,
                "type": "abstract",
                "title": "Abstract",
                "text": parsed_article.get("abstract"),
            }
        )
        for idx, section in enumerate(parsed_article["sections"], start=1):
            contents.append(
                {
                    "id": idx,
                    "type": "paragraph",
                    "title": section.get("heading", ""),
                    "text": section.get("text", ""),
                }
            )
    except Exception as e:

        traceback.print_exc(file=sys.stdout)

        title = "Error parsing PDF"
        contents = [
            {
                "id": 0,
                "type": "abstract",
                "title": "Abstract",
                "text": "Sorry, we might have problem parsing a given PDF",
            }
        ]

    return contents

def main():

    # Side bar
    st.sidebar.markdown(f"## Configuration")
    model_selected = st.sidebar.selectbox("Select Model", ("CLAIM2", "CLAIM3A", "CLAIM3B"))

    # Content
    uploaded_file = st.file_uploader("Upload PDF file")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        hash_pdf_content = hashlib.sha224(bytes_data).hexdigest()

        cache_pdf = get_cache_pdf()

        if hash_pdf_content in cache_pdf.keys():
            pdf_contents = cache_pdf[hash_pdf_content]
        else:
            # call API
            pdf_contents = parse_pdf(bytes_data)
            cache_pdf[hash_pdf_content] = pdf_contents

        for i, pdf_content in enumerate(pdf_contents):

            if i >= 5:
                break

            md = """##### Type: {} Title: {}""".format(pdf_content["type"], pdf_content["title"])

            st.markdown(md)
            st.code(pdf_content["text"])
            st.markdown("-----")

    txt = st.text_area("Enter text", "", height=250)
    btn = st.button("Predict")

    if btn:
        predictor = load_predictor()
        print("Model: {}".format(model_selected.lower()))
        result = predictor.predict(model_selected.lower(), txt)
        print(json.dumps(result, indent=2))

        display_result(result)


@st.cache(allow_output_mutation=True)
def get_cache_pdf():
    RESULT_PDF = {}

    return RESULT_PDF

@st.cache(allow_output_mutation=True)
def load_predictor():
    predictor = SCOREPredictor()
    predictor.load_model()

    return predictor

if __name__ == "__main__":
    print("Main")
    load_predictor()
    main()