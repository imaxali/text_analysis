import re
from lxml import etree


class Tokenization:
    def tokenize(self, fn):
        try:
            xml_tree = etree.parse(fn)
        except OSError as e:
            print(e)
            return 'Format to xml first'
        text_content = etree.tostring(xml_tree.getroot(), encoding='utf8', method='text').decode('utf-8')

        sentences = re.findall(r'[^\s][^.!?]+?(?:\.+|[!?])[\'"]{,2}(?=\s*(?:[A-ZА-ЯЁ]|$))', text_content)

        # r'\s+|[^\w]|(?<=[\s.,!?])\d+(?=\s\w|[^\w]{2})|\.+(?=\s*(?:[A-ZА-ЯЁ]|$)|-(?=\w+))'

        words_by_sentence = []
        for s in sentences:
            words_by_sentence.append(list(filter(lambda el: el != '', re.split(r'\s+|(?:\.+|[!?,])', s))))

        single_words = \
            list(
                filter(
                    bool,
                    re.split(
                        r'[^\aА-ЯЁа-яё]',
                        text_content
                    )
                )
            )

        return single_words

