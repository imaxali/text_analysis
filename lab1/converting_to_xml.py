import re
import codecs
from html import escape
import io
from lxml import etree, html
import docx
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pathlib import Path


class XMLConverter:
    def convert(self, fn=None, dirn=None):
        global sample_name, lab, fl

        def extract_file_data(lns, dlm):
            chapters = re.split(dlm + "{3,}", lns)
            for chapter in chapters:
                chapter = chapter.strip()
                div = etree.SubElement(doc, 'block')

                paragraphs = re.split(delim + "{1,2}", chapter)
                for i in range(len(paragraphs)):
                    pgph = etree.SubElement(div, 'block')
                    pgph.text = paragraphs[i]

        if fn is None:
            print('Type in filename')
            fn = input()
            fl = fn
            sample_name = ''
            lab = 'lab1'
        else:
            fl = Path(dirn) / fn
            sample_name = ''.join(filter(bool, re.split(r'/|\w(?!\w*/$)', dirn))) + '/'
            lab = re.match(r'\w+(?=/)', dirn).group(0)

        route = re.split(r'/', fn)
        xml_fn = '.xml'.join(re.split(r'\.\w+$', route[len(route) - 1]))

        doc = etree.Element('doc')
        doc.attrib['name'] = xml_fn

        if re.search(r'\.txt$', fn):
            f = open(fl, encoding='utf8')
            lines = ''.join(f.readlines())
            delim = r'\n'
            extract_file_data(lines, delim)

        elif re.search(r'\.html$', fn):
            file = codecs.open(fl, 'r')
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
        elif re.search(r'\.docx$', fn):
            file = docx.Document(fl)
            lines = []
            for p in file.paragraphs:
                lines.append(p.text)
            lines = '\n'.join(lines)
            delim = r'\n'
            extract_file_data(lines, delim)
        elif re.search(r'\.pdf$', fn):
            rsc_mngr = PDFResourceManager()
            fh = io.StringIO()
            converter = TextConverter(rsc_mngr, fh)
            pg_interp = PDFPageInterpreter(rsc_mngr, converter)

            fp = open(fl, 'rb')
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
        tree.write("%s/xml_samples/%s%s" % (lab, sample_name, xml_fn),
                   pretty_print=True,
                   xml_declaration=True,
                   encoding='UTF-8')
        return '%s/xml_samples/%s%s' % (lab, sample_name, xml_fn)


if __name__ == '__main__':
    XMLConverter().convert()
