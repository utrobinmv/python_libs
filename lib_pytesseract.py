import pytesseract

def find_text_card_ru_en(image_orig):
    temp = pytesseract.image_to_string(image_orig, lang='rus+eng')
    return temp
    

def find_text_boxes_ru_en(image_orig):
    boxes = pytesseract.image_to_boxes(image_orig, lang='rus+eng')
    return boxes
    