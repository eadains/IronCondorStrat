.PHONY: init requirements predict features data

# Make virtualenv, install dependencies, and initialize local git repository. Designed to be run first thing in a new project.
init:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
	git init

# Create pip requirements file
requirements:
	pip freeze > requirements.txt

# Create prediction using trained model
predict: features
	python3 src/models/predict_model.py data/processed reports

# Create processed data from raw data using /src/features/build_features.py
features: data
	python3 src/features/build_features.py data/raw data/processed

# Create raw data using src/data/make_dataset.py
data:
	python3 src/data/make_dataset.py data/raw