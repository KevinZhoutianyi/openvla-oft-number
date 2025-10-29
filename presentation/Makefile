# Makefile for generating presentation

.PHONY: all figures pdf clean

all: figures pdf

figures:
	@echo "Generating all figures..."
	python3 generate_figures.py

pdf: figures
	@echo "Compiling LaTeX presentation..."
	pdflatex presentation.tex
	pdflatex presentation.tex
	@echo "✓ Presentation compiled successfully!"
	@echo "  Output: presentation.pdf"

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
	rm -rf figures/

distclean: clean
	rm -f presentation.pdf

