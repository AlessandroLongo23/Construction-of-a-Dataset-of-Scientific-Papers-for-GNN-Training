import json
import os
import re
from pylatexenc.latex2text import LatexNodes2Text


class LatexData:
    def __init__(self, folder_path):
        self.content_tree = Tree()
        self.folder_path = folder_path

        self.tex_text = self.load_latex_file()
        self.biblio_text = self.load_bibliography_file()
        self.tex_content = self.tex_text

        self.extract_content()
        self.export_to_json()

    def export_to_json(self):
        self.content_tree.export_to_json(os.path.join(self.folder_path, 'output', 'latex_data.json'))

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
        self.replace_input_lines()
        self.remove_comments()
        self.replace_command_definitions()
        self.remove_useless_commands()

        self.content_tree.insert("root", "doc", "document", self.tex_content)
        self.extract_title()
        self.extract_authors()
        self.extract_abstract()
        self.extract_body()
        self.extract_biblio()

        self.process_leaves()
        
        print('Latex file processing done')

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

    def remove_comments(self):
        comment_pattern = r'(?m)^\%.*\n?'
        self.tex_content = re.sub(comment_pattern, '', self.tex_content, flags=re.MULTILINE) 

        comment_pattern = r'(?<!\\)%.*?$'
        self.tex_content = re.sub(comment_pattern, '', self.tex_content, flags=re.MULTILINE)

        comment_pattern = r'\\begin{comment}.*?\\end{comment}'
        self.tex_content = re.sub(comment_pattern, '', self.tex_content, flags=re.DOTALL)

    def replace_command_definitions(self):
        command_keywords = [
            'newcommand',
            'renewcommand',
            'DeclareMathOperator',
        ]
        command_definitions = []
        for command_keyword in command_keywords:
            command_pattern = r'\\(?:' + command_keyword + r')\{\\([\w\\]+)\}(?:\[(\d+)\])?\{'
            a = re.findall(command_pattern, self.tex_content)
            for b in a:
                command_definitions.append((command_keyword, b[0], b[1]))

        a = []
        for command_definition in command_definitions:
            k = r'\\' + command_definition[0] + r'\{\\' + command_definition[1] + r'\}'
            if command_definition[2] != '':
                k += r'\[' + command_definition[2] + r'\]' 
            end = re.search(k, self.tex_content).end()
            index = end + 1
            bracket_count = 1
            while bracket_count > 0:
                if self.tex_content[index] == '{':
                    bracket_count += 1
                elif self.tex_content[index] == '}':
                    bracket_count -= 1
                index += 1

            a.append((command_definition[1], command_definition[2], self.tex_content[end + 1:index - 1]))
        
        command_definitions = a
        command_dict = {name: (int(args) if args else 0, content) for name, args, content in command_definitions}

        for cmd_name, (num_args, content) in command_dict.items():
            usage_pattern = '\\' + cmd_name
            if num_args == 0:
                self.tex_content = self.tex_content.replace(usage_pattern, content)
            else:
                usage_pattern = '\\' + cmd_name + '{'
                while True:
                    start_index = self.tex_content.find(usage_pattern)
                    if start_index == -1:
                        break
                    
                    i = start_index + len(usage_pattern) - 1
                    for param_index in range(1, num_args + 1):
                        bracket_count = 0
                        end_index = None

                        while end_index is None:
                            if self.tex_content[i] == '{':
                                bracket_count += 1
                            elif self.tex_content[i] == '}':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_index = i
                            i += 1

                        content = content.replace('#' + str(param_index), self.tex_content[start_index + len(usage_pattern):end_index])
                        
                    self.tex_content = self.tex_content[:start_index] + content + self.tex_content[end_index + 1:]

    def remove_useless_commands(self):
        patterns = [
            '\\noindent',
            '\\appendix',
            '\\item',
        ]
        for pattern in patterns:
            self.tex_content = self.tex_content.replace(pattern, '')

        patterns = [
            'textbf',
            'texttt',
            'textit',
            'cite',
            'citet',
        ]
        for pattern in patterns:
            self.tex_content = re.sub(r'\\' + pattern + r'\{(.*?)\}', lambda match: match.group(1), self.tex_content)

        keywords = [
            'label',
            'autoref',
        ]
        for keyword in keywords:
            pattern = r'\\' + keyword + r'\{.*?\}'
            self.tex_content = re.sub(pattern, '', self.tex_content)

        patterns = [
            'widetext',
            'minipage',
            'wrapfigure',
        ]
        for pattern in patterns:
            self.tex_content = re.sub(r'\\begin\{' + pattern + r'\}(.*?)\\end\{' + pattern + r'\}', lambda match: match.group(1), self.tex_content, flags=re.DOTALL)

    def extract_title(self):
        title_pattern = r'\\title\s*(\[[^\]]*\])?\s*\{\s*([^}]*(\n[^}]*)*)\s*\}'
        title_match = re.search(title_pattern, self.tex_content, re.DOTALL)
        if title_match:
            title_text = title_match.group(0)
            title_content = title_match.group(2).strip()
            self.content_tree.insert("doc", "doc/tit", "title", title_content)
            self.tex_content = self.tex_content.replace(title_text, '', 1)

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
                        self.content_tree.insert("doc", "doc/aut" + str(author_found), "author", author_block[len('\\author{'):-1])
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

                author_block = self.tex_content[sorted_matches_info[0][0]:closing_index + 1]
                self.content_tree.insert("doc", "doc/aut" + str(author_found), "author", author_block[len('\\author{'):-1])
                author_found += 1
                self.tex_content = ''.join(self.tex_content.split(author_block))

                author_found = None

    def extract_abstract(self):
        abstract_pattern = r'\\begin{abstract}(.*?)\\end{abstract}'
        abstract = re.findall(abstract_pattern, self.tex_content, re.DOTALL)
        if abstract:
            self.content_tree.insert("doc", "doc/abs", "abstract", abstract[0])
            self.tex_content = re.sub(r'\\begin{abstract}.*?\\end{abstract}', '', self.tex_content, flags=re.DOTALL)

    def extract_body(self):
        body_pattern = r'\\begin{document}(.*?)\\end{document}'
        document = re.findall(body_pattern, self.tex_content, re.DOTALL)
        if document:
            self.content_tree.insert("doc", "doc/body", "body", document[0])
            self.extract_children("doc/body", document[0])

    def extract_children(self, parent_key, tex_content):
        block_index = 0
        while True:
            block_type, block_title, block_container, block_content, leaf = self.extract_next_child(tex_content)
            if not block_type:
                break

            this_key = f"{parent_key}/{block_type}{block_index}"
            self.content_tree.insert(parent_key, this_key, block_type, block_content)
            if block_title:
                self.content_tree.insert(this_key, f"{this_key}/tit", 'title', block_title)

            if leaf is False:
                self.extract_children(this_key, block_content)
            tex_content = tex_content.replace(block_container, '')
            block_index += 1

    def extract_next_child(self, tex_content):
        section_pattern = r'\\section\{([^{}]+)(?:\\label\{([^{}]+)\})?\}|\\section\{(?:[^{}]*\{[^{}]*\}[^{}]*)*\}'
        subsection_pattern = r'\\subsection\{([^{}]+)(?:\\label\{([^{}]+)\})?\}|\\subsection\{(?:[^{}]*\{[^{}]*\}[^{}]*)*\}'
        subsubsection_pattern = r'\\subsubsection\{([^{}]+)(?:\\label\{([^{}]+)\})?\}|\\subsubsection\{(?:[^{}]*\{[^{}]*\}[^{}]*)*\}'
        paragraph_pattern = r'\\paragraph\{([^{}]+)(?:\\label\{([^{}]+)\})?\}|\\paragraph\{(?:[^{}]*\{[^{}]*\}[^{}]*)*\}'
        subparagraph_pattern = r'\\subparagraph\{([^{}]+)(?:\\label\{([^{}]+)\})?\}|\\subparagraph\{(?:[^{}]*\{[^{}]*\}[^{}]*)*\}'
        text_line_pattern = re.compile(r'^(?![\s\$\\]).*?(?=(?:\n\s*\n|\n\\|\Z))', re.DOTALL | re.MULTILINE)
        begend_pattern = re.compile(r'\\begin\{([^{}]+)\}', re.DOTALL | re.MULTILINE)
        caption_pattern = r'\\caption\{'
        formula_pattern = re.compile(r'\$\$.*?\$\$', re.DOTALL | re.MULTILINE)

        def find_start_indexes(tex_content):
            patterns = {
                'section': re.compile(r'\\section\{'),
                'subsection': re.compile(r'\\subsection\{'),
                'subsubsection': re.compile(r'\\subsubsection\{'),
                'paragraph': re.compile(r'\\paragraph\{'),
                'subparagraph': re.compile(r'\\subparagraph\{'),
                'begend': re.compile(r'\\begin\{'),
                'formula': re.compile(r'\$\$'),
                'caption': re.compile(r'\\caption\{'),
                'text_line': re.compile(r'^(?![\s\$\\]).+?(?=(?:\n\s*\n|\n\\|\Z))', re.DOTALL | re.MULTILINE)
            }

            matches = {}
            for key, pattern in patterns.items():
                matches[key] = [(m.start(), key) for m in re.finditer(pattern, tex_content)]

            all_matches = []
            for match_list in matches.values():
                all_matches.extend(match_list)
            all_matches.sort()

            return all_matches

        def get_element_content(tex_content, start_index, pattern, element):
            block_end_index = tex_content.find(rf"\{element}{{", start_index + 1)
            block_end_index = block_end_index if block_end_index != -1 else len(tex_content)
                    
            container = tex_content[start_index:block_end_index]
            content = re.sub(pattern, '', container, count=1).strip()
            return container, content

        elements_with_indexes = find_start_indexes(tex_content)
        if not elements_with_indexes:
            return [None] * 5
        
        leaf = True if len(elements_with_indexes) == 1 else False

        block_start_index, corresponding_element = min(elements_with_indexes)
        if corresponding_element == 'section':
            container, content = get_element_content(tex_content, block_start_index, section_pattern, 'section')
            block_title = re.findall(section_pattern, container)[0][0]
            return 'sec', block_title, container, content, leaf

        elif corresponding_element == 'subsection':
            container, content = get_element_content(tex_content, block_start_index, subsection_pattern, 'subsection')
            block_title = re.findall(subsection_pattern, container)
            if block_title:
                return 'sub', block_title[0][0], container, content, leaf

        elif corresponding_element == 'subsubsection':
            container, content = get_element_content(tex_content, block_start_index, subsubsection_pattern, 'subsubsection')
            block_title = re.findall(subsubsection_pattern, container)[0][0]
            return 'ssb', block_title, container, content, leaf
        
        elif corresponding_element == 'paragraph':
            container, content = get_element_content(tex_content, block_start_index, paragraph_pattern, 'paragraph')
            block_title = re.findall(paragraph_pattern, container)[0][0]
            return 'par', block_title, container, content, leaf
        
        elif corresponding_element == 'subparagraph':
            container, content = get_element_content(tex_content, block_start_index, subparagraph_pattern, 'subparagraph')
            block_title = re.findall(subparagraph_pattern, container)[0][0]
            return 'sbp', block_title, container, content, leaf
        
        elif corresponding_element == "formula":
            matches = re.findall(formula_pattern, tex_content)
            if matches:
                block_end_index = tex_content.find('$$', block_start_index + 1)
                if block_end_index == -1:
                    print(f"ERROR: Cannot find closing tag for {matches[0]}")
                    return [None] * 5
                
                container = tex_content[block_start_index:block_end_index] + "$$"
                content = container.replace("$$", '')
                return "frm", None, container, content, True

        elif corresponding_element == 'caption':
            matches = re.findall(caption_pattern, tex_content)
            if matches:
                content_start_index = block_start_index + len('\caption{') 
                bracket_count = 1
                i = content_start_index
                closing_index = None

                while closing_index is None:
                    if i < len(tex_content):
                        if tex_content[i] == '{':
                            bracket_count += 1
                        elif tex_content[i] == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                closing_index = i
                    else:
                        closing_index = i - 1

                    i += 1

                container = tex_content[block_start_index:closing_index + 1]
                content = container[len('\caption{'):-1]

                return 'cpt', None, container, content, leaf

        elif corresponding_element == 'begend':
            matches = re.findall(begend_pattern, tex_content)
            if matches:
                block_end_index = tex_content.find(r'\end{' + matches[0] + r'}')
                if block_end_index == -1:
                    print(f"ERROR: Cannot find closing tag for {matches[0]}")
                    return [None] * 5
                
                container = tex_content[block_start_index:block_end_index] + f'\\end{{{matches[0]}}}'
                content = container.replace(f'\\begin{{{matches[0]}}}', '').replace(f'\\end{{{matches[0]}}}', '').strip()
                
                if content.startswith("["):
                    bracket_count = 0
                    i = 0
                    closing_index = None

                    while closing_index is None:
                        if content[i] == '[':
                            bracket_count += 1
                        elif content[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                closing_index = i
                        i += 1
                    
                    content = content[closing_index + 1:]
                
                return matches[0], None, container, content, leaf

        elif corresponding_element == 'text_line':
            if len(elements_with_indexes) == 1:
                container = tex_content[block_start_index:]
            else:
                container = tex_content[block_start_index:elements_with_indexes[1][0]]
            content = container.strip()
            return 'txl', None, container, content, leaf

        return [None] * 5
        
    def extract_biblio(self):
        if self.biblio_text is None:
            return

        bibitem_pattern = r'\\bibitem\s*(?:\[[^\]]*\])?\s*\{[^}]*\}(.*?)(?=\\bibitem|\\end\{thebibliography\})'
        bibitems = re.findall(bibitem_pattern, self.biblio_text, re.DOTALL)
        
        for i, bibitem in enumerate(bibitems):
            bibitem = bibitem.strip()
            self.content_tree.insert("doc", f"doc/bib{i}", "bibliography", bibitem)

    def process_leaves(self):
        leaves = self.get_leaves()
        for i, leaf in enumerate(leaves):
            leaf.content = self.remove_commands(leaf)
            leaf.content = re.sub(r'\s+', ' ', leaf.content).strip()

            replacements = [
                ('‘', "'"),
                ('’', "'"),
                ('`', "'"),
                ('-', ""),
                ('\\_', "_"),
            ]
            for replacement in replacements:
                leaf.content = leaf.content.replace(replacement[0], replacement[1])       
            
            leaf.leftover = leaf.content
            leaf.id = i

    def remove_commands(self, leaf):
        patterns = [
            'caption',
            'text',
            'author',
            'email',
            'affiliation',
            'emph',
            'mathrm',
            'mathbf',
            'footnote',
        ]
        
        text = leaf.content
        for pattern in patterns:
            text = re.sub(r'\\' + pattern + r'\{(.*?)\}', lambda match: match.group(1), text)

        patterns = [
            '\n',
        ]
        for pattern in patterns:
            text = re.sub(r'' + pattern, ' ', text)

        text = text.replace('\\newblock', '')
        text = text.replace('\\{', '{')
        text = text.replace('\\}', '}')
        text = text.replace('\\@', '@')
        text = re.sub(r'{\\em (.*?)}', r'\1', text)
        text = re.sub(r'{([A-Z]+)}', r'\1', text)

        def translate_equations(text):
            return LatexNodes2Text().latex_to_text(text)
        
        equation_blocks = [
            r'\$\$(.*?)\$\$',
            r'\$(.*?)\$',
            r'\\\[(.*?)\\\]',
            r'\\\((.*?)\\\)',
        ]

        for equation_block in equation_blocks:
            text = re.sub(equation_block, lambda match: translate_equations(match.group(0)), text)

        unicode_pattern = r'\\u[0-9a-fA-F]{4}'
        text = re.sub(unicode_pattern, lambda match: translate_equations(match.group(0)), text)

        text = re.sub(r'\\href\{.*?\}\{(.*?)\}', lambda match: match.group(1), text)
        text = re.sub(r'\\url\{(.*?)\}', lambda match: match.group(1), text)

        equation_block_types = [
            'algorithmic',
            'equation',
            'align',
            'gather',
            'cases',
            'frm',
        ]

        if any(block_type in leaf.block_type for block_type in equation_block_types):
            text = translate_equations(text)

        return text


class TreeNode:
    def __init__(self, parent_key, key, block_type, content):
        self.parent_key = parent_key
        self.key = key
        self.block_type = block_type
        self.id = None
        self.content = content
        self.children = []
        self.matches = []

    def to_dict(self):
        if len(self.children) == 0:
            return {
                "leaf id": self.id,
                "key": self.key,
                "block type": self.block_type,
                "content": self.content,
                "leftover": self.leftover,
                "matches": self.matches
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
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(tree_dict, json_file, indent=4, ensure_ascii=False)
