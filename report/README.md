# Dissertation Report

This directory contains the LaTeX source files for the MSc dissertation.

# Latex Packages

Run the following command to install all required packages

```bash
sudo tlmgr install slantsc relsize algorithms algorithmicx cleveref subfigure lipsum multirow makecell enumitem glossaries nomencl tocbibind appendix todonotes texcount
```

# Compiling the Thesis

The following are the minimum required settings needed to compile the `thesis.tex` with [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) in vscode:
```json
"latex-workshop.latex.tools": [
    {
        "name": "pdflatex",
        "command": "pdflatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOC%"
        ],
        "env": {}
    },
    {
        "name": "bibtex",
        "command": "bibtex",
        "args": [
            "%DOCFILE%"
        ],
        "env": {}
    },
    {
        "name": "makeindex",
        "command": "makeindex",
        "args": [
            "%DOCFILE%.nlo",
            "-s",
            "nomencl.ist",
            "-o",
            "%DOCFILE%.nls"
        ],
        "env": {}
    },
    {
        "name": "makeglossaries",
        "command": "makeglossaries",
        "args": [
            "%DOCFILE%"
        ],
        "env": {}
    },
    {
        "name": "countwords",
        "command": "texcount",
        "args": [
            "-sum",
            "-1",
            "-merge",
            "chapters/introduction.tex",
            "chapters/background.tex",
            "chapters/methodology.tex",
            "chapters/results.tex",
            "chapters/discussion.tex",
            "chapters/conclusion.tex",
            "-out=wordcount.tex"
        ],
        "env": {}
    }
],
"latex-workshop.latex.recipes": [
    {
        "name": "pdflatex ➞ makeglossaries ➞ makeindex ➞ bibtex ➞ countwords ➞ pdflatex`×2",
        "tools": [
            "pdflatex",
            "makeglossaries",
            "makeindex",
            "bibtex",
            "countwords",
            "pdflatex",
            "pdflatex"
        ]
    }
]
```