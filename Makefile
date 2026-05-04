.PHONY: help install data simulation ablation luad external figures reproduce-all clean test

help:
	@echo "GR-SAFS reproducibility targets:"
	@echo "  install         Install dependencies and the grsafs package"
	@echo "  data            Download and preprocess all cohorts"
	@echo "  simulation      Run the 3-scenario simulation benchmark"
	@echo "  ablation        Run component ablation experiments"
	@echo "  luad            Train GR-SAFS on TCGA-LUAD and select top-20 signature"
	@echo "  external        Frozen-signature validation on GSE31210 + GSE50081"
	@echo "  figures         Regenerate all paper figures"
	@echo "  reproduce-all   Run everything end-to-end"
	@echo "  test            Run unit tests"
	@echo "  clean           Remove all generated outputs"

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python scripts/00_download_data.py
	python scripts/01_preprocess_tcga.py

simulation:
	python scripts/02_run_simulation.py --jobs $(or $(JOBS),4) --seeds 20

ablation:
	python scripts/02_run_simulation.py --ablation --seeds 20

luad:
	python scripts/03_train_luad.py --output outputs/luad/

external:
	python scripts/04_external_validation.py --signature outputs/luad/signature.csv

figures:
	python scripts/05_make_figures.py

reproduce-all: install data simulation ablation luad external figures
	@echo "All paper experiments reproduced. Outputs in outputs/."

test:
	pytest tests/ -v --cov=grsafs

clean:
	rm -rf outputs/ data/processed/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
