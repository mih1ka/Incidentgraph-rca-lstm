# IncidentGraph demo

Interactive Streamlit demo for IncidentGraph (LSTM-enhanced multimodal root-cause
localization, extending Eadro). Lets you pick a model configuration and a fault
origin service, inject a simulated fault into the SocialNetwork dependency graph,
and watch detection + root-cause localization play out -- with the ranking driven
by the project's real, verified benchmark numbers (HR@1 / HR@3 per config).

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Deploy (for the live review)

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Go to https://share.streamlit.io, sign in with GitHub, "New app".
3. Pick this repo / branch, set main file to `app.py`, deploy.
4. You get a public `*.streamlit.app` link -- share it with the team and reviewers.

Benchmark numbers (F1, HR@1, HR@3, NDCG@3 per config) are hardcoded in `app.py`
from the verified SocialNetwork results (3 seeds x 5 configs). Update them there
once TrainTicket results land.
