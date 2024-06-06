import os
from WebScraper import WebScraper
from LatexData import LatexData
from PDFData import PDFData
from MatchingTool import MatchingTool

if __name__ == "__main__":
    download_folder = 'arxiv_downloads'

    # web_scraper = WebScraper('unarXive_230324_open_subset')
    # web_scraper.download_arxiv_papers("2404.15266", "2404.15270", download_folder)
    # web_scraper.reorganize_unarXive_papers()
    # web_scraper.download_unarXive_papers()

    start = 32
    end = 32
    paper_range = [i for i in range(start, end + 1)]
    for folder_name in os.listdir(download_folder):
        folder_path = os.path.join(download_folder, folder_name)
        if os.path.isdir(folder_path) and int(folder_path[-2:]) in paper_range:
            print(f'Processing {folder_path}:\n')
            os.makedirs(os.path.join(folder_path, 'output'), exist_ok=True)
            latexData = LatexData(folder_path)
            pdfData = PDFData(folder_path)
            matchingTool = MatchingTool(pdfData, latexData, folder_path)