"""Rasterize the first page of a PDF figure to PNG.

Usage: python pdf2png.py in.pdf out.png [dpi]
"""

import sys

import pymupdf

src, dst = sys.argv[1], sys.argv[2]
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 200
with pymupdf.open(src) as doc:
    doc[0].get_pixmap(dpi=dpi).save(dst)
