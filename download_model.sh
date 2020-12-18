echo "Downloading model distilRoBERTa"
gsutil cp -r gs://sentiment_analysis_yotta/distilroberta-base/ ./model/

echo "Downloading model RoBERTa"
gsutil cp -r gs://sentiment_analysis_yotta/roberta-base/ ./model/