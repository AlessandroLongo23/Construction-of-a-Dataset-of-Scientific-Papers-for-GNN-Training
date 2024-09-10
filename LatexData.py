import json
import os
import re
import regex as regex
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
                if any(r'\begin{document}' in line.strip() for line in file):
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
        self.regroup_in_one_file()
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
        
        # print('Latex file processing done')

    def regroup_in_one_file(self):
        def add_style_files():
            sty_files = [f for f in os.listdir(self.folder_path) if f.endswith('.sty')]
            for sty_file in sty_files:
                sty_file_path = os.path.join(self.folder_path, sty_file)
                with open(sty_file_path, 'r', encoding='utf-8') as file:
                    try:
                        content = file.read()
                    except UnicodeDecodeError:
                        return f"% Unable to decode file: {sty_file_path}"

                    self.tex_content = content + self.tex_content

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
            
        while True:
            input_pattern = r'\\input\*?\{([^\}]+)\}'
            match = re.search(input_pattern, self.tex_content)
            if match is None: break

            self.tex_content = re.sub(input_pattern, replace_input, self.tex_content)

        while True:
            input_pattern = r'\\include\*?\{([^\}]+)\}'
            match = re.search(input_pattern, self.tex_content)
            if match is None: break

            self.tex_content = re.sub(input_pattern, replace_input, self.tex_content)

        # add_style_files()

    def remove_comments(self):
        comment_pattern = r'\n(?<!\\)%.*$'
        self.tex_content = re.sub(comment_pattern, '', self.tex_content, flags=re.MULTILINE)
        
        for comment_pattern in [r'(?<!\\)%.*?$', r'\\begin{comment}.*?\\end{comment}']:
            self.tex_content = re.sub(comment_pattern, '', self.tex_content, flags=re.DOTALL | re.MULTILINE)

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
            open_index, close_index = self.extract_brackets_content(self.tex_content, end)
            a.append((command_definition[1], command_definition[2], self.tex_content[open_index + 1:close_index - 1]))
        
        command_definitions = a
        command_dict = {name: (int(args) if args else 0, content) for name, args, content in command_definitions}

        for cmd_name, (num_args, content) in command_dict.items():
            if num_args == 0:
                usage_pattern = r'\\' + cmd_name + r'[^\w\s]'
                while True:
                    match = re.search(usage_pattern, self.tex_content)
                    if match is None:
                        break

                    self.tex_content = self.tex_content[:match.start()] + content + ' ' + self.tex_content[match.end() - 1:]
            else:
                usage_pattern = '\\' + cmd_name + '{'
                while True:
                    start_index = self.tex_content.find(usage_pattern)
                    if start_index == -1:
                        break
                    
                    i = start_index + len(usage_pattern) - 1
                    for param_index in range(1, num_args + 1):
                        _, close_index = self.extract_brackets_content(self.tex_content, i)
                        content = content.replace('#' + str(param_index), self.tex_content[start_index + len(usage_pattern) - 1:close_index])

                    self.tex_content = self.tex_content[:start_index] + content + ' ' + self.tex_content[close_index:]

    def remove_useless_commands(self):
        text = self.tex_content
        special_characters_pattern = r'\\[ \'\`\"\^\"\~c\=.uv.Hv]*\{\w\}|\\[\`]\w' 
        text = re.sub(special_characters_pattern, lambda match: LatexNodes2Text().latex_to_text(match.group(0)), text)

        patterns = [
            '\\noindent ', 
            '\\noindent', 
            '\\appendix', 
            '\\item', 
            '\\medskip', 
            '\\bf', 
            '\\maketitle', 
            '\\hskip', 
            '\\vskip', 
            '\\smallskip ', 
            '\\relax'
        ]
        for pattern in patterns:
            text = text.replace(pattern, '')

        patterns = [
            'textbf', 
            'texttt', 
            'textit', 
            'textsf', 
            'underline', 
            'centerline',
        ]
        for pattern in patterns:
            while True:
                match = re.search(r'\\' + pattern + r'\{', text)
                if match is None: break

                open_index, close_index = self.extract_brackets_content(text, match.end() - 1)
                if text[close_index] == ' ': close_index += 1
                text = text[:match.start()] + text[open_index + 1:close_index - 1] + text[close_index:]

        citation_patterns = [
            'citet',
            'autoref', 
        ]
        for citation_pattern in citation_patterns:
            while True:
                match = re.search(r'~\\' + citation_pattern + r'\{', text)
                if match is None: break

                _, close_index = self.extract_brackets_content(text, match.end() - 1)
                if text[close_index] == ' ': close_index += 1
                text = text[:match.start()] + text[close_index:]

            while True:
                match = re.search(r'\\' + citation_pattern + r'\{', text)
                if match is None: break

                _, close_index = self.extract_brackets_content(text, match.end() - 1)
                if text[close_index] == ' ': close_index += 1
                text = text[:match.start()] + text[close_index:]

        # pattern = r'\$\\eqref\{'
        # while True:
        #     match = re.search(pattern, text)
        #     if not match: break

        #     _, close_index = self.extract_brackets_content(text, match.end() - 1, "}$")
        #     text = text[:match.start()] + text[close_index:]

        for pattern in ['widetext', 'minipage', 'wrapfigure']:
            text = re.sub(r'\\begin\{' + pattern + r'\}(.*?)\\end\{' + pattern + r'\}', lambda match: match.group(1), text, flags=re.DOTALL)

        self.tex_content = text

    def extract_title(self):
        title_match = re.search(r'\\title\{', self.tex_content, re.DOTALL)
        if not title_match:
            print("ERROR: Title not found.")
            return
        
        open_index, close_index = self.extract_brackets_content(self.tex_content, title_match.end() - 1)
        title_text = self.tex_content[title_match.start():close_index]
        title_content = self.tex_content[open_index + 1:close_index - 1]
        self.content_tree.insert("doc", "doc/tit", "title", title_content)
        self.tex_content = self.tex_content.replace(title_text, '', 1)

    def extract_authors(self):
        author_found = 0
        while True:
            matches = list()
            blocks = [
                'author',
                'affiliation',
                'orcid',
                'email',
                'institute',
                'authorrunning',
            ]
            for block in blocks:
                matches += list(re.finditer(r'\\' + block, self.tex_content))

            if len(matches) == 0:
                return

            all_matches_sorted = sorted(matches, key=lambda match: match.start())
            sorted_matches_info = [(match.start(), match.group()) for match in all_matches_sorted]

            author_block_start = sorted_matches_info[0][0]
            for i, match_info in enumerate(sorted_matches_info):
                if i > 0 and match_info[1] == "\\author":
                    break
            else:
                i += 1

            if i == len(sorted_matches_info) - 1:
                author_block_end = sorted_matches_info[i][0]
            else:
                _, author_block_end = self.extract_brackets_content(self.tex_content, sorted_matches_info[-1][0])

            author_block = self.tex_content[author_block_start:author_block_end]
            self.content_tree.insert("doc", "doc/aut" + str(author_found), "author", author_block[len('\author{'):])
            self.tex_content = self.tex_content.replace(author_block, '', 1) 
            author_found += 1

    def extract_abstract(self):
        begin_pattern, end_pattern = r'\\begin{abstract}', r'\\end{abstract}'
        begin, end = re.search(begin_pattern, self.tex_content), re.search(end_pattern, self.tex_content)
        if not begin or not end:
            print("ERROR: Abstract not found.")
            return
        
        self.content_tree.insert("doc", "doc/abs", "abstract", self.tex_content[begin.end():end.start()])
        self.tex_content = self.tex_content[:begin.start()] + self.tex_content[end.end():]

    def extract_body(self):
        begin_pattern, end_pattern = r'\\begin{document}', r'\\end{document}'
        begin, end = re.search(begin_pattern, self.tex_content), re.search(end_pattern, self.tex_content)
        if not begin or not end:
            print("ERROR: Document not found.")
            return
        
        self.content_tree.insert("doc", "doc/body", "body", self.tex_content[begin.end():end.start()])
        self.extract_children("doc/body", self.tex_content[begin.end():end.start()])

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

            equation_block_types = ['algorithmic', 'equation', 'align', 'gather', 'cases', 'frm']
            if leaf is False and not any(block_type in parent_key.split('/')[-1] for block_type in equation_block_types):
                self.extract_children(this_key, block_content)
            tex_content = tex_content.replace(block_container, '')
            block_index += 1

    def extract_next_child(self, tex_content):
        def find_start_indexes(tex_content):
            patterns = {
                'section': re.compile(r'\\section\*?\{'),
                'subsection': re.compile(r'\\subsection\*?\{'),
                'subsubsection': re.compile(r'\\subsubsection\*?\{'),
                'paragraph': re.compile(r'\\paragraph\*?\{'),
                'subparagraph': re.compile(r'\\subparagraph\*?\{'),
                'begend': re.compile(r'\\begin\{'),
                'formula$': re.compile(r'\$\$'),
                'formula[': re.compile(r'\\\['), 
                'caption': re.compile(r'\\caption\*?\{'),
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

        def get_element_content(tex_content, block_start_index, element, extract_title):
            pattern = rf"\{element}"
            block_end_index = tex_content.find(pattern, block_start_index + 1)
            if block_end_index == -1:
                block_end_index = len(tex_content)

            container = tex_content[block_start_index:block_end_index]
            pattern = rf"\\{element}"
            match = re.search(pattern, container)

            open_index, close_index = self.extract_brackets_content(container, match.end())

            if extract_title:
                title = container[open_index + 1:close_index - 1]
                content = container[close_index:].strip()
            else:
                title = None
                content = container[open_index + 1:close_index - 1] + ' ' + container[close_index:].strip()

            return container, title, content
        
        elements_with_indexes = find_start_indexes(tex_content)
        if not elements_with_indexes:
            return [None] * 5
        
        leaf = True if len(elements_with_indexes) == 1 else False

        block_start_index, corresponding_element = min(elements_with_indexes)
        document_structure_commands = [
            ('section', 'sec', True),
            ('subsection', 'sub', True),
            ('subsubsection', 'ssb', True),
            ('paragraph', 'par', False),
            ('subparagraph', 'sbp', False),
        ]
        for dsc in document_structure_commands:
            if corresponding_element != dsc[0]: continue
            
            container, block_title, content = get_element_content(tex_content, block_start_index, corresponding_element, dsc[2])
            return dsc[1], block_title, container, content, leaf
            
        if corresponding_element == "formula$":
            open_index, close_index = self.extract_brackets_content(tex_content, block_start_index, '$$', '$$')
            if open_index is None or close_index is None:
                print(f"ERROR: Cannot find opening or closing tag for {corresponding_element}")
                return [None] * 5
            
            container = tex_content[open_index:close_index]
            content = container[2:-2]
            return "frm", None, container, content, True
        
        if corresponding_element == "formula[":
            open_index, close_index = self.extract_brackets_content(tex_content, block_start_index, r'\[', r'\]')
            if open_index is None or close_index is None:
                print(f"ERROR: Cannot find opening or closing tag for {corresponding_element}")
                return [None] * 5
            
            container = tex_content[open_index:close_index]
            content = container[2:-2]
            return "frm", None, container, content, True

        if corresponding_element == 'caption':
            open_index, close_index = self.extract_brackets_content(tex_content, block_start_index)
            container = tex_content[block_start_index:close_index]
            content = tex_content[open_index + 1:close_index - 1]
            return 'cpt', None, container, content, leaf

        if corresponding_element == 'begend':
            open_index, close_index = self.extract_brackets_content(tex_content, block_start_index)
            block_type = tex_content[open_index + 1:close_index - 1]
            end_pattern = r'\\end\{' + block_type[:-1] + r'\*\}' if block_type[-1] == "*" else r'\\end\{' + block_type + r'\}'
            begend_close = re.search(end_pattern, tex_content)

            if begend_close is None:
                print(f"ERROR: Cannot find closing tag for {block_type}")
                return [None] * 5
            
            container = tex_content[block_start_index:begend_close.end()]
            content = tex_content[close_index:begend_close.start()]
            
            if content.startswith('['):
                _, close_index = self.extract_brackets_content(content, 0, '[', ']')
                content = content[close_index:]

            leaf = any(equation_block_type in block_type for equation_block_type in ['algorithmic', 'equation', 'align', 'gather', 'cases', 'tabular'])
            return block_type, None, container, content, leaf

        if corresponding_element == 'text_line':
            block_end_index = len(tex_content) if len(elements_with_indexes) == 1 else elements_with_indexes[1][0]
            container = tex_content[block_start_index:block_end_index]
            content = container.strip()
            return 'txl', None, container, content, True

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
                ('\\_', ""),
                ('_', ""),
                (' ,', ','),
            ]
            for replacement in replacements:
                leaf.content = leaf.content.replace(replacement[0], replacement[1])       
            
            leaf.leftover = leaf.content
            leaf.id = i

    def remove_commands(self, leaf):
        patterns = [
            'author', 'email', 'orcid', 'affiliation',
            'caption', 'footnote',
            'text', 'textit', 'texttt',
            'emph',
            'mathrm', 'mathbf',
            'resizebox'
        ]
        
        text = leaf.content
        for pattern in patterns:
            while True:
                match = re.search(r'\\' + pattern + r'\{', text)
                if match is None:
                    break

                open_index, close_index = self.extract_brackets_content(text, match.end() - 1)
                text = text[:match.start()] + text[open_index + 1:close_index - 1] + text[close_index:]

        inst_pattern = r'\\inst\{([\d,]+)\}'
        text = re.sub(inst_pattern, lambda match: match.group(1), text)

        symbols_replacements = [
            ('\\_', '_'),
            ('\\{', '{'),
            ('\\}', '}'),
            ('\\@', '@'),
            ('\\#', '#'),
            ('\\&', '&'),
            ('\\%', '%'),
            ('\\"', '"'),
            ('\\iid', 'i.i.d.'),
            ('\"', "''"),
        ]
        for symbol_replacement in symbols_replacements:
            text = text.replace(symbol_replacement[0], symbol_replacement[1])

        for pattern in [r'\$(.*?)\$', r'\\\((.*?)\\\)']:
            text = re.sub(pattern, lambda match: LatexNodes2Text().latex_to_text(match.group(0)), text)

        text = re.sub(r'\\u[0-9a-fA-F]{4}', lambda match: LatexNodes2Text().latex_to_text(match.group(0)), text)

        patterns = [
            (r'\\and', ', '),
            (r'\n', ' '),
            (r'\\newblock', ''),
            (r'\\\\', ' '),
            (r'\\newline', ' '),
            (r'\\centering', ''),
            (r'\\toprule', ''),
            (r'\\midrule', ''),
            (r'\\bottomrule', ''),
            (r'\\hline', ' '),
            (r'frac', ''),
            (r'\\hfil', ''),
            (r'\\hskip', ''),
            (r'\\vskip', ''),
            (r'\\linewidth', ''),
            (r'\\textwidth', ''),
            (r'\\cmark', '✓'),
            (r'\\xmark', '✗'),
            (r'\\dag', '†'),
            (r'\\ddag', '‡'),
            (r'\\textdagger', '†'),
            (r'\\textdaggerdbl', '‡'),
            (r'\\degree', '◦'),
        ]
        for pattern in patterns:
            text = re.sub(pattern[0], pattern[1], text)

        citation_patterns = [
            'cite',
            'citep',
            'citealt',
            'citealp',
            'citealpnum',
            'citeauthor',
            'citeyear',
            'citeyearpar',
            'citefullauthor',
            'citetext',
            'citenum',
            'citeonline',
            'label', 
            'cref', 
            'tableref', 
            'figureref', 
            'eqref', 
            'pageref', 
            'ref'
        ]
        for citation_pattern in citation_patterns:
            while True:
                match = re.search(r'~\\' + citation_pattern + r'\{', text)
                if match is None: break

                _, close_index = self.extract_brackets_content(text, match.end() - 1)
                text = text[:match.start()] + text[close_index:]

            while True:
                match = re.search(r'\\' + citation_pattern + r'\{', text)
                if match is None: break

                _, close_index = self.extract_brackets_content(text, match.end() - 1)
                text = text[:match.start()] + text[close_index:]

        while True:
            href_match = re.search(r'\\href', text)
            if not href_match: break

            href_start = href_match.start()
            link_url_start, link_url_end = self.extract_brackets_content(text, href_match.end())
            link_text_start, link_text_end = self.extract_brackets_content(text, link_url_end)

            text = text[:href_start] + text[link_text_start + 1:link_text_end - 1] + text[link_text_end:]

        while True:
            url_match = re.search(r'\\url', text)
            if not url_match: break

            url_start = url_match.start()
            link_url_start, link_url_end = self.extract_brackets_content(text, url_match.end())

            text = text[:url_start] + text[link_url_start + 1:link_url_end - 1] + text[link_url_end:]

        equation_block_types = ['algorithmic', 'equation', 'align', 'gather', 'cases', 'frm']
        if any(block_type in leaf.block_type for block_type in equation_block_types):
            text = LatexNodes2Text().latex_to_text(text)

        patterns = [
            # r'{\\em (.*?)}', 
            # r'{\\it (.*?)}', 
            r'{([a-zA-Z]+)}'
        ]
        for pattern in patterns:
            text = re.sub(pattern, lambda match: match.group(1), text)

        return text
    
    def extract_brackets_content(self, text, i, open_bracket='{', close_bracket='}'):
        start = None
        end = None
        bracket_count = 0
        j = i
        while end is None and i + len(open_bracket) <= len(text):
            if start is None:
                if text[i:i + len(open_bracket)] == open_bracket:
                    bracket_count += 1
                    if start is None: start = i
                    i += len(open_bracket)
                elif text[i:i + len(close_bracket)] == close_bracket:
                    bracket_count -= 1
                    i += len(close_bracket)
                    if bracket_count == 0: end = i
                else:
                    i += 1
            else:
                if text[i:i + len(close_bracket)] == close_bracket:
                    bracket_count -= 1
                    i += len(close_bracket)
                    if bracket_count == 0: end = i
                elif text[i:i + len(open_bracket)] == open_bracket:
                    bracket_count += 1
                    if start is None: start = i
                    i += len(open_bracket)
                else:
                    i += 1

        return start, end


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
