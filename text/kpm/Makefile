SUFFIX   = pdf aux log dvi ps out

NAME = kpm

all: ${NAME}.pdf

%.pdf: %.tex
	pdflatex $<

clean:
	rm -rf ${foreach suff, ${SUFFIX}, ${NAME}.${suff}}