import pytesseract

def find_text_card_ru_en(image_orig, lang='rus+eng'):
    temp = pytesseract.image_to_string(image_orig, lang=lang)
    return temp
    

def find_text_boxes_ru_en(image_orig, lang='rus+eng'):
    boxes = pytesseract.image_to_boxes(image_orig, lang=lang)
    return boxes
    