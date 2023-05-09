import os
import streamlit as st
import pytesseract # Optical Character Recognition (OCR) tool
import cv2 # OpenCV library for image processing
from tempfile import TemporaryDirectory # Used to create a temporary directory to store files
from pathlib import Path # Used to handle file paths
from pdf2image import convert_from_path # Used to convert PDF to image
from PIL import Image # Used to handle image files
import spacy # Natural language processing library
from PIL import Image, ImageEnhance, ImageFilter
import fitz
import pandas as pd
import numpy as np
from io import StringIO
import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime

st.title('Document Validator and Entity Extractor')

def file_selector(folder_path='/users/s0s0m0o/Juypter_Notebooks/'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

PDF_file = Path(filename)

# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     # To read file as bytes:
#     bytes_data = uploaded_file.getvalue()
#     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#     string_data = stringio.read()
#     # st.write(bytes_data)



import streamlit as st
import pandas as pd
import numpy as np



def getTimeStamp(pdf_date_str):
    # Remove the "D:" prefix
    date_str = pdf_date_str[2:]

    # Extract the year, month, day, hour, minute, and second from the date string
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(date_str[8:10])
    minute = int(date_str[10:12])
    second = int(date_str[12:14])

    # Calculate the timezone offset in minutes
    offset_sign = 1 if date_str[14] == '+' else -1
    offset_hours = int(date_str[15:17])
    offset_minutes = int(date_str[18:20])
    offset = datetime.timedelta(hours=offset_hours, minutes=offset_minutes) * offset_sign

    # Create a datetime object from the extracted values and the timezone offset
    # dt = datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.timezone(offset))
    dt = datetime.datetime(year, month, day, hour, minute, second)

    # Format the datetime object as a string in the desired format
    human_readable_str = dt.strftime("%d/%b/%Y %H:%M:%S %Z")

    return human_readable_str

def validate_pdf_metadata(filepath):
    with fitz.open(filepath) as doc:
        metadata = doc.metadata

    print("Metadata: "+str(metadata))
    print("Format:   "+metadata.get("format"))
    input_format = str(metadata.get("format"))
    print("Author:   "+metadata.get("author"))
    print("Creator:  "+metadata.get("creator")) 
    print("Producer: "+metadata.get("producer"))
    print("Title:    "+metadata.get("title"))
    print("Creation Date: "+metadata.get("creationDate"))
    print("Modified Date: "+metadata.get("modDate"))


    Dict = {'Format': metadata.get("format"), 'Author': metadata.get("author"), 'Creator': metadata.get("creator"), 'Producer': metadata.get("producer"), 'Creation Date':getTimeStamp(metadata.get("creationDate")), 'Modified Date':getTimeStamp(metadata.get("modDate"))}
#     df = pd.DataFrame(
#    Dict, 
#    columns=('column %d' % i for i in range(2))) 

    st.write(pd.DataFrame({
        'Values': Dict
    }))
    # st.dataframe(Dict)  # Same as st.write(df)
    # df = pd.DataFrame(columns=('Values'))
    # st.dataframe(Dict)  # Same as st.write(df) 
    # st.write(Dict)
    # st.table(Dict)
    # Check for author, creator, producer, and title metadata fields
    if metadata.get("creator") and metadata.get("producer") and metadata.get("creationDate") in metadata.get("modDate"): 
        return True
    else:
        return False




def getFileFormat(filepath):
    with fitz.open(filepath) as doc:
        metadata = doc.metadata
        return metadata.get("format");

validation_result = validate_pdf_metadata(PDF_file)
if(validation_result):
    st.header('Validation result: :green[Successful]')
else:
    st.header('Validation result: :red[Fail]')
st.write('(Validation will be successful only if value contains for Creator, Producer and Creation Date same as Modified Date)'

)
input_format = getFileFormat(PDF_file)

#Converting the PDF to images and storing them in a list
image_file_list = []
text = ""
if("pdf" in input_format.lower()):
    pdf_pages = convert_from_path(PDF_file, 500)
    image_file_list = []
    print("PDF cont..")
    for page_enumeration, page in enumerate(pdf_pages, start=1):
        # Create a file name to store the image
        filename = f"/users/s0s0m0o/Juypter_Notebooks/page_{page_enumeration:03}.jpg"
        page.save(filename, "JPEG")
        image_file_list.append(filename)
    #Extracting text from each image and printing it
    for image_file in image_file_list:
        # Converting image to text
        text = str(((pytesseract.image_to_string(Image.open(image_file)))))
        # Replacing "-\n" with "" to remove hyphenation in text
        text = text.replace("-\n", "")
        # print(text)
        # st.write(text)
else:
    # Grayscale, Gaussian blur, Otsu's thresholdz
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    # Perform text extraction
    text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
    print(text)
    st.write(text)

# remove newline characters
text = text.replace('\n', ' ')

# concatenate lines
text = ' '.join(text.split())

print(text)

@st.cache_resource
class QAMT0:
    MODEL_CHECKPOINT = "bigscience/mt0-large"
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_CHECKPOINT)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_CHECKPOINT)
        
    def extract(self, text:str, entity:str, iterate:bool=False) -> []:
        inputs  = self.tokenizer.encode("Question – {} – Q : {} ".format(text, entity), return_tensors="pt")
        outputs = self.model.generate(inputs)
        result = self.tokenizer.decode(outputs[0])
        result = result.replace("<pad>", "").replace("</s>", "").strip()
        values = result.split(', ')

        results  = []
        new_text = text
        for value in values:
            value = value.strip()
            if len(value) > 0:
                if value not in new_text:
                    # Found a hallucination
                    iterate = False
                    break
                results.append(value)
                new_text = new_text.replace(value, "").replace("  ", " ").replace(" ,", ",").replace(" .", ".").replace("() ", "").replace(" ()", "")
                
        if iterate:
            results += self.extract(new_text, entity, iterate)
        return results

qamt0 = QAMT0()

#https://pub.towardsai.net/zero-shot-ner-with-llms-ca9fad931fe0
# TEXT = "Reliance Retail Limited 5th Floor, 89, A-1 Towers, DR. Radhakrishna Salai, Mylapore, Chennai Chennai Tamil Nadu 600004 (Original for Recipient) Tax Invoice Invoice No : 33R2219991507044 Invoice/Payment Date & Time : 22 Nov,2021 13:47:01 PAN No : AABCR1718E GST No : 33AABCR1718E1ZW Order Ref. No. : TBOO0016SNEO Payment Ref. No. : 182667673472 Mode of Payment : UPI Customer Name : Sathiyajith Babu S M Place of Supply : 33 Tamil Nadu Jio Number : 4479676433 Customer Address : A14 105, A14 105,SSM NAGAR New PERUNGALATHUR, ALAPAKKAM Tambaram Tamil Nadu 600127 Sr. , , Taxable ao“ ee Piscean) | amount’ ® Platform Services - JioFiber_1M_999 998439 140.42 000 | 11800 Total Taxable Amount | 999.00 Total Amount( ®) | 1178.82 One Thousand One Hundred Seventy Eight Rupees And Eighty Two Paise Only Total Amount (in words) Telecommunication services to be provided by Reliance Jio Infocomm Limited Platform services to be provided by Jio Platforms Limited All disputes are subjected to Mumbai Jurisdiction Tax is not payable under Reverse Charge basis for this supply. Declaration : Certified that all the particulars given above are true and correct Digitally signed by DS RELIANCE RETAIL LIMITED Date: 2022.02.19 21:13:03 IST Reason: Invoice Location: Digital Signature Registered Office: Reliance Retail Limited 3rd floor, Court House, Lokmanya Tilak Marg, Dhobi Talao, Mumbai - 400002 CIN: U01100MH1999PLC120563 www.relianceretail.com"

# @st.cache_data
# def getPersonName(_text, _entity):
#         return qamt0.extract(_text, "Person Entity", iterate=True)

# @st.cache_data
# def getCost(_text):
#         return qamt0.extract(_text, "Cost", iterate=True)

# @st.cache_data
# def getDate(_text):
#         return qamt0.extract(_text, "Date", iterate=True)


title = st.text_input('Enter the entity you want to retrieve from the document')
st.write('Found Items:', qamt0.extract(text, title, iterate=True))


# def getAnything(_text, _entity):
#         return qamt0.extract(_text, _entity, iterate=True)


# st.write(getAnything(text, "Name"))
# st.write(getAnything(text, "Amount"))
# st.write(getAnything(text, "Date"))
 








































#Ok Tested
#In this updated solution, we're using the fitz module from the PyMuPDF library to read the metadata of the PDF file. The metadata is stored in a dictionary format, and we're checking if the author, creator, producer, and title metadata fields are present or not. If all four fields are present, we're returning True, indicating that the metadata is valid. If any of the fields are missing, we're returning False, indicating that the metadata is invalid.

# global input_format
# def validate_pdf_metadata(filepath):
#     with fitz.open(filepath) as doc:
#         metadata = doc.metadata

#     print("Metadata: "+str(metadata))
#     print("Format:   "+metadata.get("format"))
#     input_format = str(metadata.get("format"))
#     print("Author:   "+metadata.get("author"))
#     print("Creator:  "+metadata.get("creator")) 
#     print("Producer: "+metadata.get("producer"))
#     print("Title:    "+metadata.get("title"))
#     print("Creation Date: "+metadata.get("creationDate"))
#     print("Modified Date: "+metadata.get("modDate"))
#     Dict = {'Format': metadata.get("format"), 'Author': metadata.get("author"), 'Creator': metadata.get("creator"), 'Producer': metadata.get("producer"), 'Creation Date':metadata.get("creationDate"), 'Modified Date':metadata.get("modDate")}
#     # df = pd.DataFrame(columns=('Values'))
#     st.dataframe(Dict)  # Same as st.write(df) 
#     st.write(Dict) 
#     st.table(Dict)
#     # Check for author, creator, producer, and title metadata fields
#     if metadata.get("author") and metadata.get("creator") and metadata.get("producer") and metadata.get("title"):
#         return "Validation Result: "+ str(True)
#     else:
#         return "Validation Result: "+ str(False)




# def getFileFormat(filepath):
#     with fitz.open(filepath) as doc:
#         metadata = doc.metadata
#         return metadata.get("format");

# st.write(validate_pdf_metadata(PDF_file))
# input_format = getFileFormat(PDF_file)

# #Converting the PDF to images and storing them in a list
# image_file_list = []
# text = ""
# if("pdf" in input_format.lower()):
#     pdf_pages = convert_from_path(PDF_file, 500)
#     image_file_list = []
#     print("PDF cont..")
#     for page_enumeration, page in enumerate(pdf_pages, start=1):
#         # Create a file name to store the image
#         filename = f"/users/s0s0m0o/Juypter_Notebooks/page_{page_enumeration:03}.jpg"
#         page.save(filename, "JPEG")
#         image_file_list.append(filename)
#     #Extracting text from each image and printing it
#     for image_file in image_file_list:
#         # Converting image to text
#         text = str(((pytesseract.image_to_string(Image.open(image_file)))))
#         # Replacing "-\n" with "" to remove hyphenation in text
#         text = text.replace("-\n", "")
#         print(text)
#         st.write(text)
# else:
#     # Grayscale, Gaussian blur, Otsu's thresholdz
#     image = cv2.imread(path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (3,3), 0)
#     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#     # Morph open to remove noise and invert image
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
#     invert = 255 - opening

#     # Perform text extraction
#     text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6')
#     print(text)
#     st.write(text)

# # text = "GS CUicednvoices\n\nInvoice\n\nFrom:\nDEMO - Sliced Invoices Order Number 12345\nSuite 5A-1204 January 25, 2016\n123 Somewhere Street January 31, 2016\nYour City AZ 12345\n\noo, Total Due $93.50\nadmin @slicedinvoices.com\n\nTo:\n\nTest Business\n\n123 Somewhere St\nMelbourne, VIC 3000\ntest@test.com\n\nWeb Design g\n\nSub Total $85.00\n$650\n$00.50\n\nANZ Bank\nACC # 1234 1234\nBSB # 4821 432\n\nPayment is due within 30 days from date of invoice. Late payment is subject to fees of 5% per month.\nThanks for choosing DEMO - Sliced Invoices | admin@slicedinvoices.com\nPage 1/1"

# # remove newline characters
# text = text.replace('\n', ' ')

# # concatenate lines
# text = ' '.join(text.split())

# print(text)


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# class QAMT0:
#     MODEL_CHECKPOINT = "bigscience/mt0-large"
    
#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_CHECKPOINT)
#         self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_CHECKPOINT)
        
#     def extract(self, text:str, entity:str, iterate:bool=False) -> []:
#         inputs  = self.tokenizer.encode("Question – {} – Q : {} ".format(text, entity), return_tensors="pt")
#         outputs = self.model.generate(inputs)
#         result = self.tokenizer.decode(outputs[0])
#         result = result.replace("<pad>", "").replace("</s>", "").strip()
#         values = result.split(', ')

#         results  = []
#         new_text = text
#         for value in values:
#             value = value.strip()
#             if len(value) > 0:
#                 if value not in new_text:
#                     # Found a hallucination
#                     iterate = False
#                     break
#                 results.append(value)
#                 new_text = new_text.replace(value, "").replace("  ", " ").replace(" ,", ",").replace(" .", ".").replace("() ", "").replace(" ()", "")
                
#         if iterate:
#             results += self.extract(new_text, entity, iterate)
#         return results

# qamt0 = QAMT0()


# #https://pub.towardsai.net/zero-shot-ner-with-llms-ca9fad931fe0
# # TEXT = "Reliance Retail Limited 5th Floor, 89, A-1 Towers, DR. Radhakrishna Salai, Mylapore, Chennai Chennai Tamil Nadu 600004 (Original for Recipient) Tax Invoice Invoice No : 33R2219991507044 Invoice/Payment Date & Time : 22 Nov,2021 13:47:01 PAN No : AABCR1718E GST No : 33AABCR1718E1ZW Order Ref. No. : TBOO0016SNEO Payment Ref. No. : 182667673472 Mode of Payment : UPI Customer Name : Sathiyajith Babu S M Place of Supply : 33 Tamil Nadu Jio Number : 4479676433 Customer Address : A14 105, A14 105,SSM NAGAR New PERUNGALATHUR, ALAPAKKAM Tambaram Tamil Nadu 600127 Sr. , , Taxable ao“ ee Piscean) | amount’ ® Platform Services - JioFiber_1M_999 998439 140.42 000 | 11800 Total Taxable Amount | 999.00 Total Amount( ®) | 1178.82 One Thousand One Hundred Seventy Eight Rupees And Eighty Two Paise Only Total Amount (in words) Telecommunication services to be provided by Reliance Jio Infocomm Limited Platform services to be provided by Jio Platforms Limited All disputes are subjected to Mumbai Jurisdiction Tax is not payable under Reverse Charge basis for this supply. Declaration : Certified that all the particulars given above are true and correct Digitally signed by DS RELIANCE RETAIL LIMITED Date: 2022.02.19 21:13:03 IST Reason: Invoice Location: Digital Signature Registered Office: Reliance Retail Limited 3rd floor, Court House, Lokmanya Tilak Marg, Dhobi Talao, Mumbai - 400002 CIN: U01100MH1999PLC120563 www.relianceretail.com"
# list = qamt0.extract(text, "Person Entity", iterate=True)
# print(list[0])
# st.write(list[0])