import os
from WebScraper import WebScraper
from LatexData import LatexData
from PDFData import PDFData
from MatchingTool import MatchingTool

if __name__ == "__main__":
    download_folder = 'arxiv_downloads'

    # web_scraper = WebScraper(download_folder)
    # web_scraper.download_arxiv_papers("2404.15243", "2404.15260")

    start = 19
    end = 66
    paper_range = [i for i in range(start, end + 1)]
    for folder_name in os.listdir(download_folder):
        folder_path = os.path.join(download_folder, folder_name)
        if os.path.isdir(folder_path) and int(folder_path[-2:]) in paper_range:
            print(f'Processing {folder_path}:\n')
            latexData = LatexData(folder_path, print=False)
            pdfData = PDFData(folder_path)
            matchingTool = MatchingTool(pdfData, latexData, folder_path)
