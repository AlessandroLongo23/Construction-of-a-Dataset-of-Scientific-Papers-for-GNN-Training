import os
import requests
import tarfile
import shutil


class WebScraper:
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def download_arxiv_papers(self, start_id, end_id):
        start_index = int(start_id.replace(".", ""))
        end_index = int(end_id.replace(".", ""))

        for paper_id in range(start_index, end_index + 1):
            paper_str = str(paper_id)
            paper_str = paper_str[:4] + "." + paper_str[4:]

            paper_folder = os.path.join(self.save_path, paper_str)
            os.makedirs(paper_folder, exist_ok=True)

            pdf_url = f"https://arxiv.org/pdf/{paper_str}"
            src_url = f"https://arxiv.org/e-print/{paper_str}"

            try:
                pdf_path = self.download_file(pdf_url, paper_folder, '.pdf', rename_to='paper.pdf')
                if not pdf_path:
                    raise Exception("PDF download failed")

                src_file_path = self.download_file(src_url, paper_folder, '.tar.gz')
                if not src_file_path:
                    raise Exception("Source file download failed")

                if not self.extract_tar_gz(src_file_path, paper_folder):
                    raise Exception("Source file extraction failed")

            except Exception as e:
                print(f"Error processing paper {paper_str}: {e}")
                shutil.rmtree(paper_folder)
                print(f"Deleted folder: {paper_folder}")

    def download_file(self, url, folder, extension, rename_to=None):
        local_filename = os.path.join(folder, url.split('/')[-1])
        if not local_filename.endswith(extension):
            local_filename += extension

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {local_filename}")

            if rename_to:
                renamed_path = os.path.join(folder, rename_to)
                os.rename(local_filename, renamed_path)
                print(f"Renamed: {local_filename} to {renamed_path}")
                return renamed_path
        except requests.RequestException as e:
            print(f"Failed to download: {url}. Error: {e}")
            return None
        return local_filename

    def extract_tar_gz(self, file_path, extract_to):
        if file_path and file_path.endswith('.tar.gz'):
            try:
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=extract_to)
                print(f"Extracted: {file_path} to {extract_to}")
            except tarfile.TarError as e:
                print(f"Failed to extract {file_path}. Error: {e}")
                return False
        return True
