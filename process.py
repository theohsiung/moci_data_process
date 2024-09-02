import pdfplumber
# import mammoth
# import pypandoc
import os
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import win32com.client as win32
import PyPDF2
import re

def read_pdf_file(file_path, full_text):
    # 初始化一個空字符串來存儲PDF文檔的所有文本
    # full_text = []

    # 使用pdfplumber打開PDF文檔
    with pdfplumber.open(file_path) as pdf:
        # 遍歷每一頁
        for page in pdf.pages:
            # 提取文本並追加到full_text列表中
            text = page.extract_text()
            if text:
                full_text.append(text)

    # 將所有頁面的文本合併為一個字符串，並以換行符分隔
    return '\n'.join(full_text)

def convert_doc_to_pdf(doc_path, pdf_path=None):
    # 啟動 Word 應用程序
    word = win32.Dispatch("Word.Application")
    word.Visible = False  # 不顯示 Word 窗口

    # 打開 .doc 檔案
    doc = word.Documents.Open(doc_path)

    # 如果未指定 pdf_path，則使用與 doc_path 相同的路徑並更改副檔名為 .pdf
    if pdf_path is None:
        pdf_path = os.path.splitext(doc_path)[0] + ".pdf"

    # 將 .doc 檔案轉換成 .pdf
    doc.SaveAs(pdf_path, FileFormat=17)  # 17 表示 PDF 格式

    # 關閉檔案
    doc.Close()
    word.Quit()

    return pdf_path

def find_and_convert_docx_to_pdf(folder_path):
    # 取得資料夾中的所有檔案名稱
    files = os.listdir(folder_path)

    # 將已存在的docx檔和pdf檔分別放入集合
    docx_files = {f[:-5] for f in files if f.endswith('.docx')}
    if not docx_files:
        docx_files = {f[:-4] for f in files if f.endswith('.doc')}
    pdf_files = {f[:-4] for f in files if f.endswith('.pdf')}

    # 找出那些有docx但沒有相對應pdf的檔案
    to_convert = docx_files - pdf_files

    for docx_file in to_convert:
        docx_path = os.path.join(folder_path, f"{docx_file}.doc")
        pdf_path = os.path.join(folder_path, f"{docx_file}.pdf")
        convert_doc_to_pdf(docx_path, pdf_path)
        print(f"Converted: {docx_file}.docx to {docx_file}.pdf")

def read_pdf_content(pdf_path):
    content = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content += page.extract_text()

    # 檢查並擷取兩個字串之間的內容
    start_str = "記錄人員"
    end_str = "下次會議須準備事項："
    
    # 使用正則表達式來匹配並擷取內容
    pattern = re.compile(re.escape(start_str) + "(.*?)" + re.escape(end_str), re.DOTALL)
    match = pattern.search(content)
    
    if match:
        # 擷取到的內容
        between_content = match.group(1).strip()
        return between_content
    else:
        # 如果沒有找到匹配的內容
        return None

def main():
    #=======================================================================================
    # 找到所有文件(.doc)並轉成PDF檔
    #=======================================================================================
    root_pth = "C:\\Users\\user\\Desktop\\data"
    f_layer = os.listdir(root_pth)
    print(f_layer)

    while True:
        file_name = input("請輸入資料夾名稱: ")
        if file_name in f_layer:
            break
        else:
            print("輸入的資料夾名稱無效，請再試一次。")
    # 如果成功輸入正確的資料夾名稱，則繼續後續的操作

    print(f"你選擇的資料夾是: {file_name}")
    file_pth = os.path.join(root_pth, file_name)
    sub1 = []
    sub1 = os.listdir(file_pth)
    trgt_folder = ', '.join([folder for folder in sub1 if '專案管理' in folder])
    file2_pth = os.path.join(file_pth, trgt_folder)
    sub2 = []
    sub2 = os.listdir(file2_pth)
    trgt_folder = ', '.join([folder for folder in sub2 if '會議記錄' in folder])
    if trgt_folder == None:
        trgt_folder = ', '.join([folder for folder in sub2 if '會議紀錄' in folder])
        if trgt_folder == None:
            print(f"something went wrong in {file2_pth}")
    # print(trgt_folder)
    file3_pth = os.path.join(file2_pth, trgt_folder)
    sub3 = []
    sub3 = os.listdir(file3_pth)
    trgt_folder = ', '.join([folder for folder in sub3 if '外部' in folder])
    if trgt_folder == None:
        print(f"something went wrong in {file3_pth}")
    file4_pth = os.path.join(file3_pth, trgt_folder)
    # print(file4_pth)
    
    find_and_convert_docx_to_pdf(file4_pth)
    
    sub4 = os.listdir(file4_pth)
    for item in sub4:
        item_pth = os.path.join(file4_pth, item)
        if os.path.isdir(item_pth):
            find_and_convert_docx_to_pdf(item_pth)
    #=======================================================================================
    # function 會將讀取到的內容放入full_text 收集專案文字內容
    pdf_contents = []

    # 遍歷 file4_pth 資料夾內所有 PDF 檔案
    for item in sub4:
        item_pth = os.path.join(file4_pth, item)
        if item.endswith('.pdf'):
            try:
                content = read_pdf_content(item_pth)
                pdf_contents.append(content)
                print(f"已讀取 {item_pth}")
            except:
                print(f"something go wrong in {item_pth}")
                continue

    # 如果有子資料夾，遍歷並處理其中的 PDF 檔案
    for item in sub4:
        item_pth = os.path.join(file4_pth, item)
        if os.path.isdir(item_pth):
            for sub_item in os.listdir(item_pth):
                sub_item_pth = os.path.join(item_pth, sub_item)
                if sub_item.endswith('.pdf'):
                    try:
                        content = read_pdf_content(sub_item_pth)
                        pdf_contents.append(content)
                        print(f"已讀取 {sub_item_pth}")
                    except:
                        print(f"something go wrong in {sub_item_pth}")
                        continue

    output_file = file_pth + '\\' + file_name + '.txt'

    # 將列表內容寫入 txt 檔案
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in pdf_contents:
            file.write(line + '\n')

    print(f"列表內容已儲存到 {output_file} 中。")

main()
