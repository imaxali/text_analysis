import re
import codecs
from html import escape
import io

from lxml import etree, html
import docx
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


doc = etree.Element('doc')
doc.attrib['name'] = 'markup.xml'

print('***Txt/Html/Docx/Pdf to Xml converter ***')
print('Type in filename extension (WITH DOT).')
print('Example: .txt')
file_ext = input()


def extract_file_data(lns, delim):
    chapters = re.split(delim + "{3,}", lns)
    for chapter in chapters:
        chapter = chapter.strip()
        block = etree.SubElement(doc, 'block')

        paragraphs = re.split(delim + "{2}|" + delim + "\t", chapter)
        for i in range(len(paragraphs)):
            p = etree.SubElement(block, 'block')
            p.text = paragraphs[i]


if file_ext == '.txt':
    file = open('format_samples/sample' + file_ext)
    lines = ''.join(file.readlines())
    delim = r'\n'
    extract_file_data(lines, delim)

elif file_ext == '.html':
    file = codecs.open('format_samples/sample' + file_ext, 'r')
    file_content = file.read()
    parser = html.HTMLParser()
    html_tree = html.parse(io.StringIO(file_content), parser)
    for b in html_tree.xpath('//div[p]'):
        block = etree.SubElement(doc, 'block')
        for idx, p in enumerate(html_tree.xpath('//div/p')):
            paragraph = etree.SubElement(block, 'block')
            p_child_text = ''
            for el in html_tree.xpath('//div/p[' + str(idx + 1) + ']/*'):
                p_inner = etree.SubElement(paragraph, 'block')
                p_inner.text = escape(el.text_content())

                p_child_text = ''.join(p_child_text.split(el.text_content())) \
                    if p_child_text \
                    else ''.join(p.text_content().split(el.text_content()))
            paragraph.text = escape(''.join(re.split(r'\n{2,}| +\n', p_child_text)))
elif file_ext == '.docx':
    file = docx.Document('format_samples/sample' + file_ext)
    lines = []
    for p in file.paragraphs:
        lines.append(p.text)
    lines = '\n'.join(lines)
    delim = r'\n'
    extract_file_data(lines, delim)
elif file_ext == '.pdf':
    rsc_mngr = PDFResourceManager()
    fh = io.StringIO()
    converter = TextConverter(rsc_mngr, fh)
    pg_interp = PDFPageInterpreter(rsc_mngr, converter)

    fp = open('format_samples/sample' + file_ext, 'rb')
    for pg in PDFPage.get_pages(fp,
                                caching=True,
                                check_extractable=True):
        pg_interp.process_page(pg)

    lines = ''.join(re.split(r'\n{2,}|\x0c', fh.getvalue()))
    converter.close()
    fh.close()

    delim = ' '
    extract_file_data(lines, delim)
else:
    print('Incorrect filename extension!')

tree = etree.ElementTree(doc)
tree.write("markup.xml",
           pretty_print=True,
           xml_declaration=True,
           encoding='UTF-8')
