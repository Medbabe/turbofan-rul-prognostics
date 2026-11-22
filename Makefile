PY := .venv/bin/python

.PHONY: all setup diagnostic cv decision test clean

all: diagnostic cv decision test

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

diagnostic:
	$(PY) -m src.diagnostic

decision:
	$(PY) -m src.decision

test:
	$(PY) -m src.test_eval

cv:
	$(PY) -m src.cv

clean:
	# reports/TEST_LOCK is kept on purpose: it records the single configuration used on the test set
	rm -rf reports/figures reports/*.md reports/*.csv reports/*.csv.gz
