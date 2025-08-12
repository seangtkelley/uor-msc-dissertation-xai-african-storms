# Dissertation Report

This directory contains the LaTeX source files for the MSc dissertation.

## Compiling the Thesis

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
],
"latex-workshop.latex.recipes": [
    {
        "name": "pdflatex ➞ makeglossaries ➞ makeindex ➞ bibtex ➞ pdflatex`×2",
        "tools": [
            "pdflatex",
            "makeglossaries",
            "makeindex",
            "bibtex",
            "pdflatex",
            "pdflatex"
        ]
    }
]
```