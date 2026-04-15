import xml.etree.ElementTree as ET
import re

def extract_text_from_docx_xml(xml_file):
    # Register namespaces
    namespaces = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
        'o': 'urn:schemas-microsoft-com:office:office',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
        'v': 'urn:schemas-microsoft-com:vml',
        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
        'w10': 'urn:schemas-microsoft-com:office:word',
        'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
        'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
        'w16': 'http://schemas.microsoft.com/office/word/2018/wordml',
        'w16cex': 'http://schemas.microsoft.com/office/word/2018/wordml/cex',
        'w16cid': 'http://schemas.microsoft.com/office/word/2016/wordml/cid',
        'w16du': 'http://schemas.microsoft.com/office/word/2023/wordml/word16du',
        'w16sdtdh': 'http://schemas.microsoft.com/office/word/2020/sdtdatahash',
        'w16se': 'http://schemas.microsoft.com/office/word/2015/wordml/symex',
    }
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find all text elements
    texts = []
    for elem in root.iter():
        # Extract text from w:t elements
        if elem.tag.endswith('}t') or elem.tag == 'w:t':
            if elem.text:
                texts.append(elem.text)
        # Also handle w:tab, w:br etc.
    return ' '.join(texts)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = '/tmp/docx_extract/word/document.xml'
    text = extract_text_from_docx_xml(xml_file)
    # Print first 5000 characters
    print(text[:5000])