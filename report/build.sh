#!/bin/sh
# Biên dịch báo cáo bằng TeX Live trong Docker: ./build.sh  → main.pdf
cd "$(dirname "$0")"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work -w /work texlive/texlive:latest \
  latexmk -xelatex -shell-escape -interaction=nonstopmode -halt-on-error -file-line-error main.tex
