#!/usr/bin/env python
# coding: utf-8

# In[17]:


#Importing necessary libraries
import pytesseract # Optical Character Recognition (OCR) tool
import cv2 # OpenCV library for image processing
from tempfile import TemporaryDirectory # Used to create a temporary directory to store files
from pathlib import Path # Used to handle file paths
from pdf2image import convert_from_path # Used to convert PDF to image
from PIL import Image # Used to handle image files
import spacy # Natural language processing library
from PIL import Image, ImageEnhance, ImageFilter


# In[20]:


#Setting up the PDF file path
path = "/users/s0s0m0o/Juypter_Notebooks/Rohith-Airtel-Dongle-April2023.pdf"
PDF_file = Path(path)


# In[21]:


#Ok Tested
#In this updated solution, we're using the fitz module from the PyMuPDF library to read the metadata of the PDF file. The metadata is stored in a dictionary format, and we're checking if the author, creator, producer, and title metadata fields are present or not. If all four fields are present, we're returning True, indicating that the metadata is valid. If any of the fields are missing, we're returning False, indicating that the metadata is invalid.
import fitz

global input_format
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
    # Check for author, creator, producer, and title metadata fields
    if metadata.get("author") and metadata.get("creator") and metadata.get("producer") and metadata.get("title"):
        return "Validation Result: "+ str(True)
    else:
        return "Validation Result: "+ str(False)

def getFileFormat(filepath):
    with fitz.open(filepath) as doc:
        metadata = doc.metadata
        return metadata.get("format");

validate_pdf_metadata(PDF_file)
input_format = getFileFormat(PDF_file)


# In[22]:


print(input_format)


# In[23]:


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
        print(text)
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


# In[24]:


print(text)


# In[25]:


# text = "GS CUicednvoices\n\nInvoice\n\nFrom:\nDEMO - Sliced Invoices Order Number 12345\nSuite 5A-1204 January 25, 2016\n123 Somewhere Street January 31, 2016\nYour City AZ 12345\n\noo, Total Due $93.50\nadmin @slicedinvoices.com\n\nTo:\n\nTest Business\n\n123 Somewhere St\nMelbourne, VIC 3000\ntest@test.com\n\nWeb Design g\n\nSub Total $85.00\n$650\n$00.50\n\nANZ Bank\nACC # 1234 1234\nBSB # 4821 432\n\nPayment is due within 30 days from date of invoice. Late payment is subject to fees of 5% per month.\nThanks for choosing DEMO - Sliced Invoices | admin@slicedinvoices.com\nPage 1/1"

# remove newline characters
text = text.replace('\n', ' ')

# concatenate lines
text = ' '.join(text.split())

print(text)


# In[8]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


# In[40]:


#https://pub.towardsai.net/zero-shot-ner-with-llms-ca9fad931fe0
# TEXT = "Reliance Retail Limited 5th Floor, 89, A-1 Towers, DR. Radhakrishna Salai, Mylapore, Chennai Chennai Tamil Nadu 600004 (Original for Recipient) Tax Invoice Invoice No : 33R2219991507044 Invoice/Payment Date & Time : 22 Nov,2021 13:47:01 PAN No : AABCR1718E GST No : 33AABCR1718E1ZW Order Ref. No. : TBOO0016SNEO Payment Ref. No. : 182667673472 Mode of Payment : UPI Customer Name : Sathiyajith Babu S M Place of Supply : 33 Tamil Nadu Jio Number : 4479676433 Customer Address : A14 105, A14 105,SSM NAGAR New PERUNGALATHUR, ALAPAKKAM Tambaram Tamil Nadu 600127 Sr. , , Taxable ao“ ee Piscean) | amount’ ® Platform Services - JioFiber_1M_999 998439 140.42 000 | 11800 Total Taxable Amount | 999.00 Total Amount( ®) | 1178.82 One Thousand One Hundred Seventy Eight Rupees And Eighty Two Paise Only Total Amount (in words) Telecommunication services to be provided by Reliance Jio Infocomm Limited Platform services to be provided by Jio Platforms Limited All disputes are subjected to Mumbai Jurisdiction Tax is not payable under Reverse Charge basis for this supply. Declaration : Certified that all the particulars given above are true and correct Digitally signed by DS RELIANCE RETAIL LIMITED Date: 2022.02.19 21:13:03 IST Reason: Invoice Location: Digital Signature Registered Office: Reliance Retail Limited 3rd floor, Court House, Lokmanya Tilak Marg, Dhobi Talao, Mumbai - 400002 CIN: U01100MH1999PLC120563 www.relianceretail.com"
list = qamt0.extract(text, "Person Entity", iterate=True)
print(list[0])


# In[38]:


#https://pub.towardsai.net/zero-shot-ner-with-llms-ca9fad931fe0
# TEXT = "Reliance Retail Limited 5th Floor, 89, A-1 Towers, DR. Radhakrishna Salai, Mylapore, Chennai Chennai Tamil Nadu 600004 (Original for Recipient) Tax Invoice Invoice No : 33R2219991507044 Invoice/Payment Date & Time : 22 Nov,2021 13:47:01 PAN No : AABCR1718E GST No : 33AABCR1718E1ZW Order Ref. No. : TBOO0016SNEO Payment Ref. No. : 182667673472 Mode of Payment : UPI Customer Name : Sathiyajith Babu S M Place of Supply : 33 Tamil Nadu Jio Number : 4479676433 Customer Address : A14 105, A14 105,SSM NAGAR New PERUNGALATHUR, ALAPAKKAM Tambaram Tamil Nadu 600127 Sr. , , Taxable ao“ ee Piscean) | amount’ ® Platform Services - JioFiber_1M_999 998439 140.42 000 | 11800 Total Taxable Amount | 999.00 Total Amount( ®) | 1178.82 One Thousand One Hundred Seventy Eight Rupees And Eighty Two Paise Only Total Amount (in words) Telecommunication services to be provided by Reliance Jio Infocomm Limited Platform services to be provided by Jio Platforms Limited All disputes are subjected to Mumbai Jurisdiction Tax is not payable under Reverse Charge basis for this supply. Declaration : Certified that all the particulars given above are true and correct Digitally signed by DS RELIANCE RETAIL LIMITED Date: 2022.02.19 21:13:03 IST Reason: Invoice Location: Digital Signature Registered Office: Reliance Retail Limited 3rd floor, Court House, Lokmanya Tilak Marg, Dhobi Talao, Mumbai - 400002 CIN: U01100MH1999PLC120563 www.relianceretail.com"
list = qamt0.extract(text, "Total Amount", iterate=True)
print(list[0])


# In[34]:


#https://pub.towardsai.net/zero-shot-ner-with-llms-ca9fad931fe0
# TEXT = "Reliance Retail Limited 5th Floor, 89, A-1 Towers, DR. Radhakrishna Salai, Mylapore, Chennai Chennai Tamil Nadu 600004 (Original for Recipient) Tax Invoice Invoice No : 33R2219991507044 Invoice/Payment Date & Time : 22 Nov,2021 13:47:01 PAN No : AABCR1718E GST No : 33AABCR1718E1ZW Order Ref. No. : TBOO0016SNEO Payment Ref. No. : 182667673472 Mode of Payment : UPI Customer Name : Sathiyajith Babu S M Place of Supply : 33 Tamil Nadu Jio Number : 4479676433 Customer Address : A14 105, A14 105,SSM NAGAR New PERUNGALATHUR, ALAPAKKAM Tambaram Tamil Nadu 600127 Sr. , , Taxable ao“ ee Piscean) | amount’ ® Platform Services - JioFiber_1M_999 998439 140.42 000 | 11800 Total Taxable Amount | 999.00 Total Amount( ®) | 1178.82 One Thousand One Hundred Seventy Eight Rupees And Eighty Two Paise Only Total Amount (in words) Telecommunication services to be provided by Reliance Jio Infocomm Limited Platform services to be provided by Jio Platforms Limited All disputes are subjected to Mumbai Jurisdiction Tax is not payable under Reverse Charge basis for this supply. Declaration : Certified that all the particulars given above are true and correct Digitally signed by DS RELIANCE RETAIL LIMITED Date: 2022.02.19 21:13:03 IST Reason: Invoice Location: Digital Signature Registered Office: Reliance Retail Limited 3rd floor, Court House, Lokmanya Tilak Marg, Dhobi Talao, Mumbai - 400002 CIN: U01100MH1999PLC120563 www.relianceretail.com"
list = qamt0.extract(text, "Statement Date:", iterate=True)
print(list)


# In[29]:


#Loading the spaCy English language model
nlp = spacy.load("en_core_web_sm")


# In[41]:


#Extracting PERSON entities from the first extracted text and printing them

print("Possible Person Name Matches:")
doc = nlp(text)
for sentence in text.split('\n'):
    extract = nlp(sentence)
    for ent in extract.ents:
        if ent.label_ == 'PERSON':
            print(ent.text)


# In[42]:


#Extracting COST entities from the first extracted text and printing them

print("Possible Cost Matches:")
doc = nlp(text)
for sentence in text.split('\n'):
    extract = nlp(sentence)
    for ent in extract.ents:
        if ent.label_ == 'MONEY'or ent.label_ == 'CARDINAL':
            print(ent.text)


# In[43]:


print("Possible Date Matches:")
doc = nlp(text)
for sentence in text.split('\n'):
    extract = nlp(sentence)
    for ent in extract.ents:
        if ent.label_ == 'DATE':
            print(ent.text)


# In[ ]:




