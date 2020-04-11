import re

from lxml import etree


xml_tree = etree.parse('./markup.xml')

text_content = etree.tostring(xml_tree.getroot(), encoding='utf8', method='text').decode('utf-8')

sentences = re.findall(r'[^\s][^.!?]+?(?:\.+|[!?])[\'"]{,2}(?=\s*(?:[A-ZА-ЯЁ]|$))', text_content)

words_by_sentence = []

for s in sentences:
    words_by_sentence.append(list(filter(lambda el: el != '', re.split(r'\s+|(?:\.+|[!?,])', s))))

single_words = list(filter(lambda el: el != '', re.split(r'\s+|[!?,]|\.+(?=\s*(?:[A-ZА-ЯЁ]|$))', text_content)))

print(single_words)
