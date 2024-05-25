from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams, LTTextBox, LTFigure, LTRect, LTComponent
from pdfminer.converter import PDFPageAggregator

from tqdm import tqdm
import json
import os
from unicodeit import replace


class PDFElement:
    def __init__(self, element, page_index):
        self.element = element
        self.page_index = page_index
        if isinstance(element, LTTextBox):
            self.content = self.process_text(element.get_text())
        self.match_index = None
        self.parent = None
        self.children = []

    def contains(self, el):
        if (
            self.page_index == el.page_index and
            self.element.bbox[0] <= el.element.bbox[0] and
            self.element.bbox[1] <= el.element.bbox[1] and
            self.element.bbox[2] >= el.element.bbox[2] and
            self.element.bbox[3] >= el.element.bbox[3]
        ):
            return True
        return False
    
    def process_text(self, text):
        return replace(text.rstrip('\n'))


class PDFData:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.input_file_path = os.path.join(self.folder_path, 'paper.pdf')
        self.elements = {
            'text_boxes': [],
            'figures': [],
            'rects': [],
            'components': [],
            'containers': []
        }

        self.extract_containers()
        self.extract_elements()
        self.export_to_json()

    def extract_containers(self):
        print('\nPDF processing:')
        with open(self.input_file_path, 'rb') as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)
            resource_manager = PDFResourceManager()
            laparams = LAParams(
                all_texts=True,
                detect_vertical=False,
                line_overlap=0.0,
                char_margin=20,
                line_margin=0.35,
                word_margin=0.1,
                boxes_flow=0.0
            )
            device = PDFPageAggregator(resource_manager, laparams=laparams)
            interpreter = PDFPageInterpreter(resource_manager, device)

            for page_index, page in enumerate(tqdm(list(PDFPage.create_pages(document)), desc=f"- Extracting blocks")):
                interpreter.process_page(page)
                layout = device.get_result()
                for element in layout:
                    el = PDFElement(element, page_index)
                    if isinstance(element, LTTextBox):
                        self.elements['containers'].append(el)
            print('')

    def extract_elements(self):
        with open(self.input_file_path, 'rb') as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)
            resource_manager = PDFResourceManager()
            laparams = LAParams(
                all_texts=True,
                detect_vertical=False,
                line_overlap=0.0,
                char_margin=20,
                line_margin=0.0,
                word_margin=0.1,
                boxes_flow=0.0
            )
            device = PDFPageAggregator(resource_manager, laparams=laparams)
            interpreter = PDFPageInterpreter(resource_manager, device)

            for page_index, page in enumerate(tqdm(list(PDFPage.create_pages(document)), desc=f"- Extracting elements")):
                interpreter.process_page(page)
                layout = device.get_result()
                for element in layout:
                    el = PDFElement(element, page_index)
                    if isinstance(element, LTFigure):
                        self.elements['figures'].append(el)

                    elif isinstance(element, LTTextBox):
                        self.elements['text_boxes'].append(el)
                        for container in self.elements['containers']:
                            if container.contains(el):
                                el.parent = container
                                container.children.append(el)

                    elif isinstance(element, LTRect):
                        self.elements['rects'].append(el)

                    elif isinstance(element, LTComponent):
                        self.elements['components'].append(el)

            print('')

    def export_to_json(self):
        def element_to_dict(el):
            return {
                'content': el.content,
                'page_index': el.page_index,
                'bbox': el.element.bbox,
                'match': el.match_index,
            }

        data = [element_to_dict(el) for el in self.elements['text_boxes']]
        
        output_file_path = os.path.join(self.folder_path, 'pdf_data.json')
        with open(output_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
