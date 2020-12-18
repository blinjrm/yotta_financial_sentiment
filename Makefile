
init: ## initiate virtual environment
	bash activate.sh

download_models: ## download models
	bash download_model.sh

black:  ## run black formater
	black .
