#.SILENT:
SHELL = /bin/bash


all:
	echo -e "Required section:\n\
 build - build project into build directory, with configuration file and environment\n\
 clean - clean all addition file, build directory and output archive file\n\
 test - run all tests\n\
 pack - make output archive, file name format \"sarima_vX.Y.Z_BRANCHNAME.tar.gz\"\n\
"

VERSION := "0.2.0"
BRANCH := $(shell git name-rev $$(git rev-parse HEAD) | cut -d\  -f2 | sed -re 's/^(remotes\/)?origin\///' | tr '/' '_')

pack: make_build
	rm -f sarima-*.tar.gz
	echo Create archive \"sarima-$(VERSION)-$(BRANCH).tar.gz\"
	cd make_build; tar czf ../sarima-$(VERSION)-$(BRANCH).tar.gz sarima

clean_pack:
	rm -f sarima-*.tar.gz


sarima.tar.gz: build
	cd make_build; tar czf ../sarima.tar.gz sarima && rm -rf ../make_build

build: make_build

make_build:
	# required section
	echo make_build
	mkdir make_build
	cp -R ./sarima make_build
	cp *.md make_build/sarima/



clean_build:
	rm -rf make_build

venv:
	echo Create venv;
	conda create --copy -p ./venv -y
	conda install -p ./venv python==3.9.7 -y
	./venv/bin/pip install --no-input  postprocessing_sdk@git+ssh://git@github.com/ISGNeuroTeam/postprocessing_sdk.git@develop

clean_venv:
	rm -rf ./venv

pp_cmd: venv
	./venv/bin/pp_sdk createcommandlinks

otl_v1_config.ini:
	echo -e "[spark]\n\
base_address = http://localhost\n\
username = admin\n\
password = 12345678\n\
\n\
[caching]\n\
# 24 hours in seconds\n\
login_cache_ttl = 86400\n\
# Command syntax defaults\n\
default_request_cache_ttl = 100\n\
default_job_timeout = 100\n\
" > $@


dev: pp_cmd otl_v1_config.ini
	ln -s -r ./sarima pp_cmd/sarima

clean: clean_build clean_pack clean_test clean_venv

test:
	@echo "Testing..."

clean_test:
	@echo "Clean tests"