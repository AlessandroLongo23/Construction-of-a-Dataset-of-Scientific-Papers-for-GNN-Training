from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams, LTTextBox, LTFigure, LTRect, LTComponent
from pdfminer.converter import PDFPageAggregator
from pylatexenc.latex2text import LatexNodes2Text

from tqdm import tqdm
from collections import defaultdict

import json
import os
import re


class PDFElement:
    def __init__(self, element, page_index, index):
        self.element = element
        self.page_index = page_index
        self.index = index
        if isinstance(element, LTTextBox):
            self.prefix = ''
            self.content = self.process_text(element.get_text())
            self.suffix = ''
        self.assigned = False
        self.matches = []
        self.equation_hit = False
        self.eq_neighbours = []
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
        def translate_equations(text):
            return LatexNodes2Text().latex_to_text(text)

        unicode_pattern = r'\\u[0-9a-fA-F]{4}'
        text = re.sub(unicode_pattern, lambda match: translate_equations(match.group(0)), text)

        text = text.replace('‘', "'")
        text = text.replace('’', "'")
        text = text.replace('`', "'")
        return text


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

        self.extract_pages()
        self.export_to_json()
        print('PDF file processing done')

    def extract_pages(self):
        with open(self.input_file_path, 'rb') as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)
            resource_manager = PDFResourceManager()

            laparams_containers = LAParams(
                all_texts=True,
                detect_vertical=False,
                char_margin=20,
                word_margin=0.1,
                line_margin=0.35,
                line_overlap=0.0,
                boxes_flow=0.0
            )
            device_containers = PDFPageAggregator(resource_manager, laparams=laparams_containers)
            interpreter_containers = PDFPageInterpreter(resource_manager, device_containers)

            laparams_elements = LAParams(
                all_texts=True,
                detect_vertical=False,
                char_margin=20,
                word_margin=0.1,
                line_margin=0.0,
                line_overlap=0.0,
                boxes_flow=0.0
            )
            device_elements = PDFPageAggregator(resource_manager, laparams=laparams_elements)
            interpreter_elements = PDFPageInterpreter(resource_manager, device_elements)

            for page_index, page in enumerate(tqdm(list(PDFPage.create_pages(document)), desc=f"Extracting PDF pages", leave=False)):
                interpreter_containers.process_page(page)
                layout_containers = device_containers.get_result()
                for index, element in enumerate(layout_containers):
                    el = PDFElement(element, page_index, index)
                    if isinstance(element, LTTextBox):
                        self.elements['containers'].append(el)

                interpreter_elements.process_page(page)
                layout_elements = device_elements.get_result()
                for index, element in enumerate(layout_elements):
                    el = PDFElement(element, page_index, index)
                    if isinstance(element, LTFigure):
                        self.elements['figures'].append(el)

                    elif isinstance(element, LTTextBox):
                        self.elements['text_boxes'].append(el)
                        for container in self.elements['containers']:
                            if container.contains(el) and el.content in container.content:
                                el.parent = container
                                container.children.append(el)
                                el.prefix, el.content, el.suffix = self.calc_context(el.content, container.content)

                    elif isinstance(element, LTRect):
                        self.elements['rects'].append(el)

                    elif isinstance(element, LTComponent):
                        self.elements['components'].append(el)

    def calc_context(self, content, container):
        context_words = 5
        start_index = container.find(content)
        end_index = start_index + len(content)

        i = start_index
        count = context_words
        while count > 0 and i >= 0:
            i -= 1
            if container[i] == ' ':
                count -= 1
        prefix = container[i + 1:start_index]

        prefix = prefix.replace('−\n', "")
        prefix = prefix.replace('-\n', "")
        prefix = prefix.replace('\n', " ")
        prefix = prefix.replace('−', "")
        prefix = prefix.replace('-', "")

        content = content.replace('−\n', "")
        content = content.replace('-\n', "")
        content = content.replace('\n', " ")
        content = content.replace('−', "")
        content = content.replace('-', "")

        i = end_index
        count = context_words
        while count > 0 and i < len(container):
            if container[i] == ' ':
                count -= 1
            i += 1
        suffix = container[end_index:i - 1]

        suffix = suffix.replace('−\n', "")
        suffix = suffix.replace('-\n', "")
        suffix = suffix.replace('\n', " ")
        suffix = suffix.replace('−', "")
        suffix = suffix.replace('-', "")

        return prefix, content, suffix

    def export_to_json(self):
        def element_to_dict(el, index):
            return {
                'page': el.page_index,
                'index': index,
                'bbox': el.element.bbox,
                # 'parent': el.parent.content,
                'prefix': el.prefix,
                'content': el.content,
                'suffix': el.suffix,
                'equation hit': el.equation_hit,
                # 'eq_neighbours': el.eq_neighbours,
                'assigned': el.assigned,
                'matches': el.matches,
            }

        page_groups = defaultdict(list)
        for el in self.elements['text_boxes']:
            page_groups[el.page_index].append(el)

        data = []
        for page_index, elements in page_groups.items():
            for index, el in enumerate(elements):
                data.append(element_to_dict(el, index))

        output_file_path = os.path.join(self.folder_path, 'output', 'pdf_data.json')

        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
