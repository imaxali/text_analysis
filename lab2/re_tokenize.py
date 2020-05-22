import re
from lxml import etree
from num2words import num2words


class Tokenization:
    def tokenize(self, txt, fn=True):
        global text_content

        def is_numeric(s):
            if s.isnumeric() is True:
                try:
                    s = num2words(s, lang='en')
                except:
                    pass
            return s

        def is_single(c):
            return '' if len(c) == 1 else c

        if fn is not True:
            text_content = txt
        else:
            try:
                xml_tree = etree.parse(txt)
            except (OSError, TypeError) as e:
                print(e)
                return 'Format to xml first'
            text_content = etree.tostring(xml_tree.getroot(), encoding='utf8', method='text').decode('utf-8')

            sentences = re.findall(r'[^\s][^.!?]+?(?:\.+|[!?])[\'"]{,2}(?=\s*(?:[A-ZА-ЯЁ]|$))', text_content)

            words_by_sentence = []
            for s in sentences:
                words_by_sentence.append(list(filter(lambda el: el != '', re.split(r'\s+|(?:\.+|[!?,])', s))))

        text_content = re.split(r'[^\w]', text_content)

        single_words = \
            list(
                filter(
                    bool,
                    map(
                        is_single,
                        map(
                            is_numeric,
                            text_content
                        )
                    )
                )
            )

        return single_words


if __name__ == '__main__':
    print(Tokenization().tokenize('Я так думаю, значит так оно и есть. 10 игроков.', False))

