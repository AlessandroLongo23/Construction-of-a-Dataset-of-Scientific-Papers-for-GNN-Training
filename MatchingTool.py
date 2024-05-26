import colorsys
import math
import fitz
import cv2
import numpy as np
import webbrowser
import os
import re
from tqdm import tqdm
from pdf2image import convert_from_path
from PIL import Image


class MatchingTool:
    def __init__(self, pdf_data, latex_data, folder_path):
        self.pdf_data = pdf_data
        self.latex_data = latex_data
        self.folder_path = folder_path
        self.threshold = [i for i in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]

        self.perform_matching()

    def perform_matching(self):
        tex_lines = self.latex_data.get_leaves()
        pdf_lines = self.pdf_data.elements['text_boxes']

        score_matrix = self.create_score_matrix(tex_lines, pdf_lines)
        first_match_pdf_path = self.first_match(tex_lines, pdf_lines, score_matrix)
        self.calculate_remainings(first_match_pdf_path)
        self.pdf_data.export_to_json()

    def create_score_matrix(self, tex_lines, pdf_lines):
        def longest_common_subsequence(a, b, equation=False):
            def is_equal(a, b, equation=False):
                a = a if equation else a.lower()
                b = b if equation else b.lower()

                if a == b:
                    return True

                if b.startswith('~\\cite{') and b.endswith('}'):
                    if a.startswith('[') and a.endswith(']'):
                        a_numbers = a[1:-1].split(',')
                        if all(num.strip().isdigit() for num in a_numbers):
                            b_citations = b[7:-1].split(',')
                            if len(a_numbers) == len(b_citations):
                                return True

                return False
        
            if a in b:
                return 1
            
            words_a = re.findall(r'\[.*?\]|[\w-]+[^\w\s]*|\S', a) if not equation else a
            words_b = b.split() if not equation else b

            m, n = len(words_a), len(words_b)
            matches = 0
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            pt = [[1] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if is_equal(words_a[i - 1], words_b[j - 1], equation):
                        dp[i][j] = dp[i - 1][j - 1] + math.exp(-(min(pt[i - 1][j], pt[i][j - 1]) - 1) / 10)
                        pt[i][j] = 0
                        matches += 1
                    else:
                        if dp[i - 1][j] > dp[i][j - 1]:
                            dp[i][j] = dp[i - 1][j]
                            pt[i][j] = pt[i - 1][j] + 1 if dp[i - 1][j] != 0 else pt[i - 1][j]
                        else:
                            dp[i][j] = dp[i][j - 1]
                            pt[i][j] = pt[i][j - 1] + 1 if dp[i][j - 1] != 0 else pt[i][j - 1]

            # lcs = []
            # i, j = m, n
            # while i > 0 and j > 0:
            #     if words_a[i - 1] == words_b[j - 1]:
            #         lcs.append(words_a[i - 1])
            #         i -= 1
            #         j -= 1
            #     elif dp[i - 1][j] > dp[i][j - 1]:
            #         i -= 1
            #     else:
            #         j -= 1
            #
            # lcs.reverse()

            return dp[m][n] / m if m > 0 else 0

        print('\nLaTex-PDF matching:')
        score_matrix = np.zeros((len(tex_lines), len(pdf_lines)))
        for i, tex_line in enumerate(tqdm(tex_lines, desc=f"- Creating score matrix")):
            for j, pdf_line in enumerate(pdf_lines):
                score_matrix[i][j] = longest_common_subsequence(pdf_line.content, tex_line.content)
        print('')

        return score_matrix
    
    def first_match(self, tex_lines, pdf_lines, score_matrix):
        def locality_bonus(pdf_lines, pdf_line, tex_line_index):
            mult = 1
            first = True
            parent = pdf_line.parent

            if parent is not None:
                distances = [abs(sibling.match_index - tex_line_index) for sibling in parent.children if sibling.match_index is not None]
                if distances:
                    mean_distance = sum(distances) / len(distances)
                    mult *= 1 / (1 + mean_distance ** 2)
                    first = False

            page_siblings = [line for line in pdf_lines if line.page_index == pdf_line.page_index and line.match_index is not None]
            if page_siblings:
                distances = [abs(sibling.match_index - tex_line_index) for sibling in page_siblings]
                mean_distance = sum(distances) / len(distances)
                mult *= 1 / (1 + 1 / (40 ** 2) * (mean_distance ** 2))
                first = False

            return 1 if first else mult
        
        for threshold in tqdm(self.threshold, desc='- First matching'):
            for j, pdf_line in enumerate(tqdm(pdf_lines, desc=f"   - confidence level: >{threshold * 100}%", leave=False)):
                if pdf_line.match_index is None:
                    best_score = -1
                    best_tex_index = None
                    for i in range(len(tex_lines)):
                        adjusted_score = score_matrix[i][j] * locality_bonus(pdf_lines, pdf_line, i)
                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_tex_index = i

                    if best_score >= threshold:
                        pdf_line.match_index = best_tex_index
        
        return self.mark_pdf()

    # def remove_lcs(self, a, b):
    #     words_a = a.split()
    #     words_b = b.split()
    #     m, n = len(words_a), len(words_b)
    #     dp = [[0] * (n + 1) for _ in range(m + 1)]
    #     pt = [[1] * (n + 1) for _ in range(m + 1)]
    #
    #     for i in range(1, m + 1):
    #         for j in range(1, n + 1):
    #             if words_a[i - 1] == words_b[j - 1]:
    #                 dp[i][j] = dp[i - 1][j - 1] + math.exp(-(min(pt[i - 1][j], pt[i][j - 1]) - 1) / 5)
    #                 pt[i][j] = 0
    #             else:
    #                 if dp[i - 1][j] > dp[i][j - 1]:
    #                     dp[i][j] = dp[i - 1][j]
    #                     if dp[i - 1][j] == 0:
    #                         pt[i][j] = pt[i - 1][j]
    #                     else:
    #                         pt[i][j] = pt[i - 1][j] + 1
    #                 else:
    #                     dp[i][j] = dp[i][j - 1]
    #                     if dp[i][j - 1] == 0:
    #                         pt[i][j] = pt[i][j - 1]
    #                     else:
    #                         pt[i][j] = pt[i][j - 1] + 1
    #
    #     without_lcs = ""
    #     i, j = m, n
    #     while i > 0 and j > 0:
    #         if words_a[i - 1] == words_b[j - 1]:
    #             i -= 1
    #             j -= 1
    #         elif dp[i - 1][j] > dp[i][j - 1]:
    #             without_lcs = words_a[i - 1] + " " + without_lcs
    #             i -= 1
    #         else:
    #             j -= 1
    #
    #     while i > 0:
    #         without_lcs = words_a[i - 1] + " " + without_lcs
    #         i -= 1
    #
    #     return without_lcs.strip()

    class ConnectedComponent:
        def __init__(self, nodes=[]):
            self.nodes = nodes
            self.calculate_bbox()

        def calculate_bbox(self):
            self.min_x, self.min_y = float('inf'), float('inf')
            self.max_x, self.max_y = -float('inf'), -float('inf')

            for node in self.nodes:
                self.min_x = min(self.min_x, node.bbox.x)
                self.min_y = min(self.min_y, node.bbox.y)
                self.max_x = max(self.max_x, node.bbox.x + node.bbox.w)
                self.max_y = max(self.max_y, node.bbox.y + node.bbox.h)

        def draw_bbox(self, img_bgr):
            cv2.rectangle(img_bgr, (self.min_x, self.min_y), (self.max_x, self.max_y), (0, 0, 0, 0.5), 1)

    class Bbox:
        def __init__(self, x, y, w, h):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
    
        def distance_to(self, bbox):
            x1, y1, w1, h1 = self.x, self.y, self.w, self.h
            x2, y2, w2, h2 = bbox.x, bbox.y, bbox.w, bbox.h
            
            left1, right1, bottom1, top1 = x1, x1 + w1, y1, y1 + h1
            left2, right2, bottom2, top2 = x2, x2 + w2, y2, y2 + h2
            
            if right1 >= left2 and left1 <= right2 and top1 >= bottom2 and bottom1 <= top2:
                return 0.0
            
            horizontal_distance = max(left2 - right1, left1 - right2, 0)
            vertical_distance = max(bottom2 - top1, bottom1 - top2, 0)
            
            if horizontal_distance > 0 and vertical_distance == 0:
                return horizontal_distance
            if vertical_distance > 0 and horizontal_distance == 0:
                return vertical_distance
            
            return (horizontal_distance ** 2 + 4 * vertical_distance ** 2) ** 0.5

    class Node:
        def __init__(self, index, bbox):
            self.index = index
            self.bbox = bbox
            self.neighbours = []

    def calculate_remainings(self, pdf_path):
        with open(pdf_path, 'rb') as f:
            pdf_document_labels = fitz.open(pdf_path)

            with open(self.pdf_data.input_file_path, 'rb') as f:
                pdf_document_remainings = fitz.open(self.pdf_data.input_file_path)

                for element_type in ['text_boxes', 'figures', 'rects', 'components', 'containers']:
                    for element in self.pdf_data.elements[element_type]:
                        self.clean_document(pdf_document_remainings, element)

                remainings_file_path = os.path.join(self.folder_path, 'remainings.pdf')
                pdf_document_remainings.save(remainings_file_path)
                pdf_document_remainings.close()
                output_folder = os.path.join(self.folder_path, 'remainings')
                os.makedirs(output_folder, exist_ok=True)
                images = convert_from_path(remainings_file_path, dpi=300)
                
            print('')
            scale = images[0].width / pdf_document_labels[0].rect.width
            for idx, image in enumerate(tqdm(images, desc=f"- find and process remainings")):
                connected_components, img_with_cc = self.find_connected_components(image)
                for cc in connected_components:
                    rect = fitz.Rect((cc.min_x / scale, cc.min_y / scale, cc.max_x / scale, cc.max_y / scale))
                    pdf_document_labels[idx].draw_rect(rect, color=(0, 0, 0, 1))
                img_with_cc.save(os.path.join(output_folder, f'page_{idx + 1}.jpg'))

            labeled_file_path = os.path.join(self.folder_path, 'labeled2.pdf')
            pdf_document_labels.save(labeled_file_path)
            pdf_document_labels.close()
            print('\nArxiv ' + labeled_file_path.split('\\')[1] + ': "' + self.latex_data.content_tree.find_node('doc/tit').content + '" processed and matched with latex')
            print('\n' + '-' * 100 + '\n')
            webbrowser.open_new_tab(labeled_file_path)

    def clean_document(self, pdf_document, el):
        if el.match_index is not None:
            page = pdf_document[el.page_index]
            x0, y0, x1, y1 = el.element.bbox
            page_height = page.mediabox[3] - page.mediabox[1]
            adjusted_y0 = page_height - y1
            adjusted_y1 = page_height - y0
            bbox = (x0, adjusted_y0, x1, adjusted_y1)
            rect = fitz.Rect(bbox)
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

    def find_connected_components(self, image):
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        nodes = []
        for i in range(1, num_labels):
            x, y, w, h, _ = stats[i]
            nodes.append(self.Node(i, self.Bbox(x, y, w, h)))

        distance_threshold = 40
        for node in nodes:
            for other_node in nodes:
                if node.index != other_node.index and node.bbox.distance_to(other_node.bbox) <= distance_threshold:
                    node.neighbours.append(other_node)
                    other_node.neighbours.append(node)

        def dfs(node, visited, component):
            stack = [node]
            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    component.append(n)
                    for neighbour in n.neighbours:
                        if neighbour not in visited:
                            stack.append(neighbour)

        visited = set()
        connected_components = []
        for node in nodes:
            if node not in visited:
                component = []
                dfs(node, visited, component)
                connected_components.append(self.ConnectedComponent(component))

        for cc in connected_components:
            cc.draw_bbox(img_bgr)

        img_with_rects = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return connected_components, Image.fromarray(img_with_rects)

    def mark_pdf(self):
        with open(self.pdf_data.input_file_path, 'rb') as f:
            pdf_document = fitz.open(self.pdf_data.input_file_path)

            num_leaves = len(self.pdf_data.elements['text_boxes'])
            for text_box in self.pdf_data.elements['text_boxes']:
                self.draw_bounding_box(pdf_document, text_box, None, num_leaves)

            for figure in self.pdf_data.elements['figures']:
                self.draw_bounding_box(pdf_document, figure, (1, 1, 0))

            for rect in self.pdf_data.elements['rects']:
                self.draw_bounding_box(pdf_document, rect, (0, 0, 1))

            for component in self.pdf_data.elements['components']:
                self.draw_bounding_box(pdf_document, component, (1, 0, 1))

            for container in self.pdf_data.elements['containers']:
                self.draw_bounding_box(pdf_document, container, (0, 0, 0, 0.5))

            output_file_path = os.path.join(self.folder_path, 'labeled.pdf')
            pdf_document.save(output_file_path)
            pdf_document.close()
            return output_file_path
            # print('\nArxiv ' + output_file_path.split('\\')[1] + ': "' + self.latex_data.content_tree.find_node('doc/tit').content + '" processed and matched with latex')
            # print('\n' + '-' * 100 + '\n')
            # webbrowser.open_new_tab(output_file_path)

    def draw_bounding_box(self, pdf_document, el, colorbox=None, num_leaves=1):
        if el.match_index is not None:
            page = pdf_document[el.page_index]
            x0, y0, x1, y1 = el.element.bbox
            page_height = page.mediabox[3] - page.mediabox[1]
            adjusted_y0 = page_height - y1
            adjusted_y1 = page_height - y0
            bbox = (x0, adjusted_y0, x1, adjusted_y1)
            rect = fitz.Rect(bbox)

            if colorbox is None:
                normalized_index = el.match_index / num_leaves
                hue = (normalized_index * num_leaves / 4.35436) % 1 * 360
                saturation = 0.8
                brightness = 0.8
                rgb_color = colorsys.hsv_to_rgb(hue / 360, saturation, brightness)
                color = (rgb_color[0], rgb_color[1], rgb_color[2])
                page.insert_text((x1 + 3, adjusted_y1), str(el.match_index), fontsize=10, color=color)

                page.draw_rect(rect, color=color)
            # else:
            #     color = colorbox

            # if el in self.elements['containers']:
            #     x, y, width, height = rect
            #     rect = (x - 1, y - 1, width + 1, height + 1)

            # page.draw_rect(rect, color=color)


def remove_unicode(a, b):
    result_a = ''
    result_b = ''
    for char_a, char_b in zip(a, b):
        if char_a.encode('ascii', 'ignore').decode('utf-8') != char_a:
            continue

        result_a += char_a
        result_b += char_b
    return result_a, result_b