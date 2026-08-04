.PHONY: setup test magnetics solve50 solve6300
setup:
	./setup.sh

test:
	python -m pytest -q

magnetics:
	python cli.py magnetics

solve50:
	python cli.py solve --freq 50 --drive current --current 1 --render

solve6300:
	python cli.py solve --freq 6300 --drive current --current 1 --render
