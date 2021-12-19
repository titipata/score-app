# SCORE Prediction

Streamlit application for [SCORE](https://www.cos.io/score) API.
To run the application locally, use the following commands:

```sh
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Models

Models were trained from a collection of claim annotation from around 3000 journals.
Claims are specified in 3 levels: abstract level (claim2), paragraph level (claim3a), and results level (claim3b).

**Note** This is a work-in-progress implementation. We only provide one model in our API.
