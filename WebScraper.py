import os
import requests
import tarfile
import shutil
import json
import re
from tqdm import tqdm


class WebScraper:
    def __init__(self, unarXive_folder):
        self.unarXive_folder = unarXive_folder

    def download_arxiv_papers(self, start_id, end_id, save_path):
        os.makedirs(save_path, exist_ok=True)

        start_index = int(start_id.replace(".", ""))
        end_index = int(end_id.replace(".", ""))

        for paper_id in range(start_index, end_index + 1):
            paper_str = str(paper_id)
            paper_str = paper_str[:4] + "." + paper_str[4:]

            paper_folder = os.path.join(self.save_path, paper_str)
            os.makedirs(paper_folder, exist_ok=True)

            pdf_url = f"https://arxiv.org/pdf/{paper_str}"
            src_url = f"https://arxiv.org/e-print/{paper_str}"

            print(f"{paper_str}")
            try:
                pdf_path = self.download_file(pdf_url, paper_folder, '.pdf', rename_to='paper.pdf')
                if not pdf_path:
                    raise Exception("PDF download failed")

                src_file_path = self.download_file(src_url, paper_folder, '.tar.gz')
                if not src_file_path:
                    raise Exception("source file download failed")

                if not self.extract_tar_gz(src_file_path, paper_folder):
                    raise Exception("source file extraction failed")
                
                print(f"Output: Paper downloaded\n" + "-" * 200)

            except Exception as e:
                print(f"[x] error processing paper: {e}")
                shutil.rmtree(paper_folder)
                print(f"Output: Paper discarded\n" + "-" * 200)

    def download_file(self, url, folder, extension, rename_to=None, print_info=False):
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
            if print_info: print(f"[-] files downloaded successfully")

            if rename_to:
                renamed_path = os.path.join(folder, rename_to)
                os.rename(local_filename, renamed_path)
                if print_info: print(f"[-] pdf renamed successfully")
                return renamed_path
        except requests.RequestException as e:
            if print_info: print(f"[x] files failed to download: {e}")
            return None
        return local_filename

    def extract_tar_gz(self, file_path, extract_to):
        if file_path and file_path.endswith('.tar.gz'):
            try:
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=extract_to)
                print(f"[-] source zip extracted")
            except tarfile.TarError as e:
                print(f"[x] source zip failed to extract: {e}")
                return False
        return True
    
    def reorganize_unarXive_papers(self):
        def search_and_process_files(current_folder):
            for root, _, files in os.walk(current_folder):
                for file in tqdm(files, desc=f'{current_folder}/processing files: ', leave=False):
                    if file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            json_objects = file_content.splitlines()
                            
                            for json_object in json_objects:
                                data = json.loads(json_object)
                                paper_id = data['paper_id']
                                paper_id_clean = re.sub(r'[\/:*?"<>|]', '_', paper_id)
                                new_file_name = f"arXiv_{paper_id_clean}.json"
                                new_file_path = os.path.join(root, new_file_name)
                                with open(new_file_path, 'w', encoding='utf-8') as json_file:
                                    json.dump(data, json_file, indent=4)

                        os.remove(file_path)
                        

        search_and_process_files(self.unarXive_folder)

    def download_unarXive_papers(self):
        def search_and_download_pdf(root):
            for current_folder, _, files in os.walk(root):
                for file in tqdm(files, desc=f'{current_folder} processing: '):
                    if file.endswith('.json'):
                        file_path = os.path.join(current_folder, file)
                        if not os.path.exists(file_path[:-5] + '.pdf'):
                            file_str = file_path.split("\\")[-1][:-5]
                            paper_str = file_str[6:].replace('_', '/')

                            pdf_url = f"https://arxiv.org/pdf/{paper_str}"
                            try:
                                pdf_path = self.download_file(pdf_url, current_folder, '.pdf', rename_to=f'{file_str}.pdf')
                                if not pdf_path:
                                    raise Exception("PDF download failed")

                                # print(f"Output: Paper downloaded\n" + "-" * 100)

                            except Exception as e:
                                print(f"[x] error processing paper: {e}")
                                print(f"Output: Paper discarded\n" + "-" * 200)


        search_and_download_pdf(self.unarXive_folder)

