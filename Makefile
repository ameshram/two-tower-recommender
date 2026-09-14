.PHONY: setup test lint eval train data train-movielens quickstart clean

setup:  ## install with dev extras
	pip install -e ".[dev]"

test:  ## run unit tests (incl. gradient check)
	pytest

lint:
	ruff check .

train:  ## train on synthetic data and print metrics vs popularity
	python -m twotower.train

eval:  ## train + evaluate + regression gates (offline synthetic)
	python -m eval.run_eval --gate

data:  ## download MovieLens 100k (opt-in; requires network)
	python scripts/download_movielens.py --out data/ml-100k

train-movielens: ## train on real MovieLens 100k (run `make data` first)
	python -m twotower.train --movielens data/ml-100k --epochs 30

quickstart: setup test eval  ## everything a reviewer needs, offline

clean:
	rm -rf data eval/report.md eval/report.json .pytest_cache *.npz
