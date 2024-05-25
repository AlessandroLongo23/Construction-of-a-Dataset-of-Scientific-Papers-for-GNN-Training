import colorsys
import math
import fitz
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import os
import re
from tqdm import tqdm


class MatchingTool:
    def __init__(self, pdf_data, latex_data, folder_path):
        self.pdf_data = pdf_data
        self.latex_data = latex_data
        self.folder_path = folder_path
        self.threshold = [i for i in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]]

        self.perform_matching()
        self.mark_pdf()

    def locality_bonus(self, pdf_lines, pdf_line, tex_line_index):
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
            mult *= 1 / (1 + 1 / 2500 * mean_distance ** 2)
            first = False

        return 1 if first else mult

    def perform_matching(self):
        tex_lines = self.latex_data.get_leaves()
        pdf_lines = self.pdf_data.elements['text_boxes']

        score_matrix = np.zeros((len(tex_lines), len(pdf_lines)))
        for i, tex_line in enumerate(tqdm(tex_lines, desc=f"- Creating score matrix")):
            for j, pdf_line in enumerate(pdf_lines):
                score_matrix[i][j] = longest_common_subsequence(pdf_line.content, tex_line.content)
        print('')

        for threshold in tqdm(self.threshold, desc='- Performing PDF-Latex matching'):
            for j, pdf_line in enumerate(tqdm(pdf_lines, desc=f"   - confidence level: {threshold * 100}%", leave=False)):
                if pdf_line.match_index is None:
                    best_score = -1
                    best_tex_index = None
                    for i, tex_line in enumerate(tex_lines):
                        adjusted_score = score_matrix[i][j] * self.locality_bonus(pdf_lines, pdf_line, i)
                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_tex_index = i

                    if best_score >= threshold:
                        pdf_line.match_index = best_tex_index

        self.pdf_data.export_to_json()

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

        # if len(text_box.children) > 0:
        #     center_x = (x0 + x1) / 2
        #     center_y = (adjusted_y0 + adjusted_y1) / 2
        #     text_box_center = (center_x, center_y)
        #     for child in text_box.children:
        #         sx0, sy0, sx1, sy1 = child.element.bbox
        #         sadjusted_y0 = page_height - sy1
        #         sadjusted_y1 = page_height - sy0
        #         child_center_x = (sx0 + sx1) / 2 + (random() - 0.5) * 20
        #         child_center_y = (sadjusted_y0 + sadjusted_y1) / 2
        #         child_center = (child_center_x, child_center_y)
        #
        #         page.draw_line(text_box_center, child_center, color=color)

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

            output_file_path = os.path.join(self.folder_path, 'paper_with_bboxes.pdf')
            pdf_document.save(output_file_path)
            pdf_document.close()
            print('\nArxiv ' + output_file_path.split('\\')[1] + ': "' + self.latex_data.content_tree.find_node('doc/tit').content + '" processed and matched with latex')
            print('\n' + '-' * 100 + '\n')
            webbrowser.open_new_tab(output_file_path)


def remove_unicode(a, b):
        result_a = ''
        result_b = ''
        for char_a, char_b in zip(a, b):
            if char_a.encode('ascii', 'ignore').decode('utf-8') != char_a:
                continue

            result_a += char_a
            result_b += char_b
        return result_a, result_b

def longest_common_subsequence(a, b, equation=False):
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