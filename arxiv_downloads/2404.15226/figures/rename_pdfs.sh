# a script to rename all pdfs {file}.pdf to {file}_pdf.pdf
# Usage: bash rename_pdfs.sh

for file in *.pdf; do
    mv "$file" "${file%.pdf}_pdf.pdf"
done

# End of file
