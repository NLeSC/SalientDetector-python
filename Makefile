init:
	conda install --file requirements.txt
	conda install -c https://conda.anaconda.org/menpo opencv3
all:
	python setup.py build
install:
	python setup.py install
clean:
	python setup.py clean
	rm -fr build
test:
	nosetests tests
