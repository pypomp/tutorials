.PHONY: install_requirements install_pypi install_git

install_pypi: install_requirements
	pip install pypomp

install_git: install_requirements
	pip install git+https://github.com/pypomp/pypomp.git

install_git_latest: install_requirements
	pip install git+https://github.com/pypomp/pypomp.git --force-reinstall --no-deps

install_requirements: .venv
	pip install -r requirements.txt

.venv:
	python3.12 -m venv .venv
