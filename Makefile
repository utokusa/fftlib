.PHONY: lint
lint:
	git ls-files | grep -e '\.py$$' | xargs pylint

.PHONY: lint-fix
lint-fix:
	git ls-files | grep -e '\.py$$' | xargs black

.PHONY: test
test:
	python3 fft.py

.PHONY: check-all
check-all:
	make lint && make test

