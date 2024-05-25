import os
from WebScraper import WebScraper
from LatexData import LatexData
from PDFData import PDFData
from MatchingTool import MatchingTool, longest_common_subsequence, is_equal
from pylatexenc.latex2text import LatexNodes2Text

if __name__ == "__main__":
    download_folder = 'arxiv_downloads'

    web_scraper = WebScraper.py(download_folder)
    web_scraper.download_arxiv_papers("2404.15243", "2404.15260")

    start = 40
    end = 40
    paper_range = [i for i in range(start, end + 1)]
    for folder_name in os.listdir(download_folder):
        folder_path = os.path.join(download_folder, folder_name)
        if os.path.isdir(folder_path) and int(folder_path[-2:]) in paper_range:
            print(f'Processing {folder_path}:\n')
            latexData = LatexData(folder_path, print=False)
            pdfData = PDFData(folder_path)
            matchingTool = MatchingTool(pdfData, latexData, folder_path)


    pdf = "Symmetric Ideals and Invariant Hilbert Schemes"
    correct = "SYMMETRIC IDEALS AND INVARIANT HILBERT SCHEMES"
   
    print(f'{longest_common_subsequence(pdf, correct)}')
