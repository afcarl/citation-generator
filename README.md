# citation-generator

Command-line tool for using a corpus to add citations to a document.

## Usage

`execute.py` is a command-line script which gives more information. To get started:

    cd ~/Desktop
    git clone https://github.com/codekansas/citation-generator
    chmod +x citation-generator/execute.py
    mkdir data/ # your documents go in here, as raw text files
    citation-generator/execute.py -d data/ -f /path/to/file # the file you want to annotate
