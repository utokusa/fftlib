#!/bin/bash

# exit when any command fails
set -e

display_usage() { 
	echo "Lint checking script for C++ codes."
    echo -e "If no argument is passed, it runs with check mode as default.\n"
	echo "Usage:"
    echo -e "$0 [check | fix]"
    echo -e "$0 -h | --help" 
    echo ""
}

# Tips: clang-format without this script
# $ git ls-files | grep -e '\.h$' -e '\.cpp$' | xargs clang-format -i --dry-run 

TARGET="all"
MODE="check"

for i in "$@"
do
    if [ "$i" == "fix" ]
    then
        MODE="fix"
    fi

    if [ "$i" == "--help" ] || [ "$i" == "-h" ]
    then
        display_usage
        exit 0
    fi
done

OS="Unknown"
if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    OS="MacOSX"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
    OS="Linux"
elif [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ]; then
    # Do something under Windows platform
    OS="Windows"
fi

if [ "$OS" == "MacOSX" ] || [ "$OS" == "Linux" ] || [ "$OS" == "Windows" ]; then
    CPP_FILES=`git ls-files | grep -e '\.h$' -e '\.cpp$'`
else
    echo "Unknow OS"
    exit 1
fi


echo "[$0] $MODE cpp files"   
if [ "$MODE" == "fix" ]; then
    clang-format -i $CPP_FILES
else
    clang-format -i --dry-run --Werror $CPP_FILES
fi
echo "[$0] OK"
