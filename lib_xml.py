import xml.etree.cElementTree as ET

def create_xml_annotation_bbox_with_brand(filename,folder_image,width,height,depth,bbox_list):
    
    root = ET.Element("annotation")
    
    ET.SubElement(root, "folder").text = folder_image
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = folder_image + '/' + filename
    
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = 'Unknown'

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    
    ET.SubElement(root, "segmented").text = str(0)
    
    for bbox in bbox_list:
        x1,y1,x2,y2,name = bbox
        
        xml_object = ET.SubElement(root, "object")
        
        ET.SubElement(xml_object, "name").text = name
        ET.SubElement(xml_object, "pose").text = 'Unspecified'
        ET.SubElement(xml_object, "truncated").text = str(0)
        ET.SubElement(xml_object, "difficult").text = str(0)
        
        bndbox = ET.SubElement(xml_object, "bndbox")
        
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)
    
    
    tree = ET.ElementTree(root)
    
    return tree


def create_xml_annotation(filename,folder_image,width,height,depth,name,bbox_list):
    
    root = ET.Element("annotation")
    
    ET.SubElement(root, "folder").text = folder_image
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = folder_image + '/' + filename
    
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = 'Unknown'

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    
    ET.SubElement(root, "segmented").text = str(0)
    
    for bbox in bbox_list:
        x1,y1,x2,y2 = bbox
        
        xml_object = ET.SubElement(root, "object")
        
        ET.SubElement(xml_object, "name").text = name
        ET.SubElement(xml_object, "pose").text = 'Unspecified'
        ET.SubElement(xml_object, "truncated").text = str(0)
        ET.SubElement(xml_object, "difficult").text = str(0)
        
        bndbox = ET.SubElement(xml_object, "bndbox")
        
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)
    
    
    tree = ET.ElementTree(root)
    
    return tree
    
def write_xml_to_file(tree, filename):
    tree.write(filename, xml_declaration=True, encoding="utf-8")
    