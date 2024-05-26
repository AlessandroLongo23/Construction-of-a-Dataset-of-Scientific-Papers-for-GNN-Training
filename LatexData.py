import json
import os
import re
from pylatexenc.latex2text import LatexNodes2Text


class LatexData:
    def __init__(self, folder_path, print=False):
        self.content_tree = Tree()
        self.folder_path = folder_path

        self.tex_text = self.load_latex_file()
        self.biblio_text = self.load_bibliography_file()
        self.tex_content = self.tex_text

        self.extract_content()
        self.content_tree.print_tree() if print is True else None
        self.content_tree.export_to_json(os.path.join(self.folder_path, 'latex_data.json'))

    def load_latex_file(self):
        tex_files = [f for f in os.listdir(self.folder_path) if f.endswith('.tex')]
        for tex_file in tex_files:
            tex_file_path = os.path.join(self.folder_path, tex_file)
            with open(tex_file_path, 'r', encoding='utf-8') as file:
                if any(line.strip() == r'\begin{document}' for line in file):
                    return self.load_file(tex_file_path)
        return None

    def load_bibliography_file(self):
        bbl_file = None
        for file in os.listdir(self.folder_path):
            if file.endswith('.bbl'):
                bbl_file = file

        if bbl_file is None:
            return None

        bbl_file_path = os.path.join(self.folder_path, bbl_file)
        return self.load_file(bbl_file_path)

    def load_file(self, file):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            print(f"Error: File '{file}' not found.")
            return None
        except Exception as e:
            print(f"Error: An unexpected error occurred: {e}")
            return None

    def find_all(self, keyword):
        return self.content_tree.find_all(self.content_tree.root, keyword)

    def get_leaves(self):
        return self.content_tree.get_leaves()

    def extract_content(self):
        print('Latex file processing:')
        print('- preprocessing:')
        self.preprocess()
        self.content_tree.insert("root", "doc", "document", self.tex_content)

        print('- content extraction:')
        self.extract_title()
        self.extract_authors()
        self.extract_abstract()
        self.extract_body()
        self.extract_biblio()
        self.process_leaves()

    def preprocess(self):
        self.replace_input_lines()
        self.remove_comments()
        self.replace_commands()

    def replace_input_lines(self):
        def replace_input(match):
            input_path = os.path.join(self.folder_path, match.group(1)) + '.tex'
            try:
                with open(input_path, 'r', encoding='utf-8') as file:
                    try:
                        content = file.read()
                    except UnicodeDecodeError:
                        return f"% Unable to decode file: {input_path}"

                    return content
            except FileNotFoundError:
                return f"% File not found: {input_path}"

        pattern = r'\\input\*?\{([^\}]+)\}'
        self.tex_content = re.sub(pattern, replace_input, self.tex_content)
        pattern = r'\\include\*?\{([^\}]+)\}'
        self.tex_content = re.sub(pattern, replace_input, self.tex_content)
        print('   - input and include commands replaced with content')

    def remove_comments(self):
        comment_pattern = r'(?<!\\)%.*?$'
        self.tex_content = re.sub(comment_pattern, '', self.tex_content, flags=re.MULTILINE)
        print('   - comments removed')

        pattern = r'\\begin{comment}.*?\\end{comment}'
        self.tex_content = re.sub(pattern, '', self.tex_content, flags=re.DOTALL)

    def replace_commands(self):
        command_pattern = r'\\(?:newcommand|renewcommand)\{\\([\w\\]+)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        command_definitions = re.findall(command_pattern, self.tex_content)
        command_dict = dict(command_definitions)
        
        def replace_command(match):
            command_name = match.group(1)
            content = command_dict.get(command_name)
            if content:
                if not match.group(0).startswith('\\'):
                    return match.group(0)
                else:
                    return content
            else:
                return match.group(0)
        
        self.tex_content = re.sub(r'\\(\w+)\b', replace_command, self.tex_content)
        print('   - commands replaced with content')

    def extract_title(self):
        title_pattern = r'\\title\s*(\[[^\]]*\])?\s*\{\s*([^}]*(\n[^}]*)*)\s*\}'
        title_match = re.search(title_pattern, self.tex_content, re.DOTALL)
        if title_match:
            title_text = title_match.group(0)
            title_content = title_match.group(2).strip()
            self.content_tree.insert("doc", "doc/tit", "title", title_content)
            self.tex_content = self.tex_content.replace(title_text, '', 1)
            print('   - title extracted')

    def extract_authors(self):
        author_found = 0
        while author_found is not None:
            author_matches = list(re.finditer(r'\\author', self.tex_content))
            affiliation_matches = list(re.finditer(r'\\affiliation', self.tex_content))
            orcid_matches = list(re.finditer(r'\\orcid', self.tex_content))
            email_matches = list(re.finditer(r'\\email', self.tex_content))

            all_matches = author_matches + affiliation_matches + orcid_matches + email_matches
            all_matches_sorted = sorted(all_matches, key=lambda match: match.start())
            sorted_matches_info = [(match.start(), match.group()) for match in all_matches_sorted]

            if not sorted_matches_info:
                break

            count = 0
            for index, match in enumerate(sorted_matches_info):
                if match[1] == "\\author":
                    count += 1
                    if count == 1:
                        author_block_start = match[0]
                    elif count == 2:
                        author_block_end = match[0]
                        author_block = self.tex_content[author_block_start:author_block_end]
                        self.content_tree.insert("doc", "doc/aut" + str(author_found), "author", author_block)
                        author_found += 1
                        self.tex_content = ''.join(self.tex_content.split(author_block))
                        sorted_matches_info = sorted_matches_info[index:]

            if count < 2:
                last_author_block = sorted_matches_info[-1]
                bracket_count = 0
                i = last_author_block[0]
                closing_index = None

                while closing_index is None:
                    if self.tex_content[i] == '{':
                        bracket_count += 1
                    elif self.tex_content[i] == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            closing_index = i
                    i += 1

                author_block = self.tex_content[sorted_matches_info[0][0]: closing_index + 1]
                self.content_tree.insert("doc", "doc/aut" + str(author_found), "author", author_block)
                author_found += 1
                self.tex_content = ''.join(self.tex_content.split(author_block))

                author_found = None

        print('   - authors extracted')

    def extract_abstract(self):
        abstract_pattern = r'\\begin{abstract}(.*?)\\end{abstract}'
        abstract = re.findall(abstract_pattern, self.tex_content, re.DOTALL)
        self.content_tree.insert("doc", "doc/abs", "abstract", abstract[0])
        self.tex_content = re.sub(r'\\begin{abstract}.*?\\end{abstract}', '', self.tex_content, flags=re.DOTALL)
        print('   - abstract extracted')

    def extract_body(self):
        body_pattern = r'\\begin{document}(.*?)\\end{document}'
        document = re.findall(body_pattern, self.tex_content, re.DOTALL)
        if document:
            self.content_tree.insert("doc", "doc/body", "body", document[0])
            self.extract_children("doc/body", document[0])
            print('   - body extracted')

    def extract_children(self, parent_key, tex_content, block_index=0):
        while True:
            block_type, block_container, block_content = self.extract_next_child(parent_key, block_index, tex_content)
            if block_type == -1:
                break
            this_key = f"{parent_key}/{block_type}{block_index}"
            self.content_tree.insert(parent_key, this_key, block_type, block_content)
            self.extract_children(this_key, block_content)
            tex_content = tex_content.replace(block_container, '')
            block_index += 1

    def extract_next_child(self, parent_key, block_index, tex_content):
        section_pattern = r'\\section\{([^{}]+)(?:\\label\{([^{}]+)\})?\}|\\section\{(?:[^{}]*\{[^{}]*\}[^{}]*)*\}'
        subsection_pattern = r'\\subsection\{([^{}]+)(?:\\label\{([^{}]+)\})?\}|\\subsection\{(?:[^{}]*\{[^{}]*\}[^{}]*)*\}'
        subsubsection_pattern = r'\\subsubsection\{([^{}]+)(?:\\label\{([^{}]+)\})?\}|\\subsubsection\{(?:[^{}]*\{[^{}]*\}[^{}]*)*\}'
        paragraph_pattern = re.compile(r'^[^\s\\].*?(?=(?:\n\s*\n|\n\\))', re.DOTALL | re.MULTILINE)
        begend_pattern = r'\\begin\{([^{}]+)\}'

        def find_start_indexes(tex_content):
            start_indexes = [
                (tex_content.find(r'\section{'), 'section'),
                (tex_content.find(r'\subsection{'), 'subsection'),
                (tex_content.find(r'\subsubsection{'), 'subsubsection'),
                (paragraph_pattern.search(tex_content).start() if paragraph_pattern.search(tex_content) else -1, 'paragraph'),
                (tex_content.find(r'\begin{'), 'begend')
            ]
            return [(idx, elem) for idx, elem in start_indexes if idx != -1]

        def get_element_content(tex_content, start_index, pattern, element):
            block_end_index = tex_content.find(rf'\\{element}\{{', start_index + 1)
            block_end_index = block_end_index if block_end_index != -1 else len(tex_content)
                    
            container = tex_content[start_index:block_end_index]
            content = re.sub(pattern, '', container, count=1).strip()
            return container, content

        elements_with_indexes = find_start_indexes(tex_content)
        if not elements_with_indexes:
            return -1, -1, -1

        block_start_index, corresponding_element = min(elements_with_indexes)
        if corresponding_element == 'section':
            container, content = get_element_content(tex_content, block_start_index, section_pattern, 'section')
            block_title = re.findall(section_pattern, container)
            self.content_tree.insert(parent_key, f"{parent_key}/sec{block_index}/tit", 'section', block_title[0][0])
            return 'sec', container, content

        elif corresponding_element == 'subsection':
            container, content = get_element_content(tex_content, block_start_index, subsection_pattern, 'subsection')
            block_title = re.findall(subsection_pattern, container)
            self.content_tree.insert(parent_key, f"{parent_key}/sub{block_index}/tit", 'subsection', block_title[0][0])
            return 'sub', container, content

        elif corresponding_element == 'subsubsection':
            container, content = get_element_content(tex_content, block_start_index, subsubsection_pattern, 'subsubsection')
            block_title = re.findall(subsubsection_pattern, container)
            self.content_tree.insert(parent_key, f"{parent_key}/ssb{block_index}/tit", 'subsubsection', block_title[0][0])
            return 'ssb', container, content

        elif corresponding_element == 'paragraph':
            par = paragraph_pattern.search(tex_content)
            if par:
                container = par.group(0)
                content = container.strip()
                return 'par', container, content

        elif corresponding_element == 'begend':
            matches = re.findall(begend_pattern, tex_content)
            if matches:
                block_end_index = tex_content.find(r'\end{' + matches[0] + r'}')
                if block_end_index == -1:
                    return -1, -1, -1
                container = tex_content[block_start_index:block_end_index] + f'\\end{{{matches[0]}}}'
                content = container.replace(f'\\begin{{{matches[0]}}}', '').replace(f'\\end{{{matches[0]}}}', '').strip()
                return matches[0], container, content

        return -1, -1, -1
        
    def extract_biblio(self):
        if self.biblio_text is None:
            return

        bibitem_pattern = r'\\bibitem\s*(?:\[[^\]]*\])?\s*\{[^}]*\}(.*?)(?=\\bibitem|\\end\{thebibliography\})'
        bibitems = re.findall(bibitem_pattern, self.biblio_text, re.DOTALL)
        
        for i, bibitem in enumerate(bibitems):
            bibitem = bibitem.strip()
            self.content_tree.insert("doc", f"doc/bib{i}", "bibliography", bibitem)

        print('   - bibliography extracted')

    def process_leaves(self):
        leaves = self.get_leaves()
        for i, leaf in enumerate(leaves):
            leaf.content = self.remove_commands(leaf.content)
            # leaf.content = self.translate_equations(leaf.content)
            leaf.id = i

    def remove_commands(self, text):
        patterns = [
            'caption',
            'text',
            'textbf',
            'author',
            'email',
            'affiliation',
            'emph',
            'mathrm'
            'mathbf'
        ]

        for pattern in patterns:
            text = re.sub(r'\\' + pattern + r'\{(.*?)\}', lambda match: match.group(1), text)

        patterns = [
            '\n',
            '\\noindent',
        ]
        for pattern in patterns:
            text = re.sub(r'' + pattern, ' ', text)
        text = text.replace('\\newblock', '')
        text = re.sub(r'{\\em (.*?)}', r'\1', text)
        text = re.sub(r'{([A-Z]+)}', r'\1', text)

        def translate_equations(text):
            return LatexNodes2Text().latex_to_text(text)

        text = re.sub(r'\$(.*?)\$', lambda match: translate_equations(match.group(0)), text)

        return text


class TreeNode:
    def __init__(self, parent_key, key, block_type, content):
        self.parent_key = parent_key
        self.key = key
        self.block_type = block_type
        self.id = None
        self.content = content
        self.children = []

    def to_dict(self):
        if len(self.children) == 0:
            return {
                "leaf id": self.id,
                "key": self.key,
                "block type": self.block_type,
                "content": self.content,
            }
        else:
            return {
                "key": self.key,
                "block_type": self.block_type,
                "children": [child.to_dict() for child in self.children],
            }


class Tree:
    def __init__(self):
        self.root = None

    def insert(self, parent_key, key, block_type, content):
        if self.root is None:
            self.root = TreeNode("root", key, block_type, content)
            return

        parent_node = self.find_node(parent_key)
        if parent_node:
            parent_node.children.append(TreeNode(parent_key, key, block_type, content))
        else:
            print("Parent node not found. Key:", parent_key)

    def print_tree(self):
        if self.root:
            self._recursive_print(self.root, 0)

    def _recursive_print(self, node, depth):
        if len(node.children) == 0:
            print("\t" * depth + f"Key: {node.key}")
            print("\t" * depth + f"Id: {node.id}")
            print("\t" * depth + f"Content: {node.content}")
            print("-" * 200)

        for child in node.children:
            self._recursive_print(child, depth + 1)

    def find_node(self, key, node=None):
        if node is None:
            node = self.root

        if node.key == key:
            return node

        for child in node.children:
            found = self.find_node(key, child)
            if found:
                return found

        return None

    def find_all(self, node, key, results=None):
        if results is None:
            results = []

        if key in node.key:
            results.append(node)

        for child in node.children:
            self.find_all(child, key, results)

        return results

    def get_leaves(self, node=None, leaves=None):
        leaves = [] if leaves is None else leaves
        node = self.root if node is None else node

        if not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                self.get_leaves(child, leaves)

        return leaves

    def to_dict(self):
        if self.root:
            return self.root.to_dict()
        return {}

    def export_to_json(self, file_path):
        tree_dict = self.to_dict()
        with open(file_path, 'w') as json_file:
            json.dump(tree_dict, json_file, indent=4)

        print('- Content tree exported as json\n')
