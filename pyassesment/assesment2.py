import csv
import json
import logging
from pathlib import Path
from typing import List,Optional,Union
import openpyxl
import fitz
#logging setting dynamic layout control
logging.basicConfig(level=logging.ERROR,format='%(levelname)s-%(message)s')
logger=logging.getLogger(__name__)
def setup_dummy_files()->None:
    csv_content="name,department,salary\narun,sales,30000\npriya,hr,45000\nrahul,it,60000\ndivya,it,75000\nkarthi,finance,90000"
    Path("employees.csv").write_text(csv_content)

    doc=fitz.open()
    page=doc.new_page()
    page.insert_text((50,50),"py file handling and environment assign content."*5)
    doc.save("sample.pdf")
    doc.close()
def process_employee_data(csv_path:Path,json_path:Path,high_earners_path:Path)-> None:
    rows_read = 0
    high_earners_written=0
    employee_list:List[dict]=[]
    high_earners_list:List[dict]=[]

    with csv_path.open(mode='r',encoding='utf-8')as f:
        reader=csv.DictReader(f)
        for row in reader:
            rows_read += 1
            row['salary']=int(row['salary'])
            employee_list.append(row)

            if row['salary'] > 50000:
                high_earners_list.append(row)
                high_earners_written += 1

    with json_path.open(mode='w',encoding='utf-8') as f:
        json.dump(employee_list,f,indent=4)
    with high_earners_path.open(mode='w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=["name","department","salary"])
        writer.writeheader()
        writer.writerows(high_earners_list)

    print(f"rows read:{rows_read} | high earners written: {high_earners_written}")
          
def manage_excel(excel_path: Path, source_csv: Path) -> None:
    wb = openpyxl.Workbook()
    ws= wb.active
    ws.title = "Salaries"
    with source_csv.open(mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            ws.append(row)
            wb.save(excel_path)
            wb_read = openpyxl.load_workbook(excel_path)
            ws_read = wb_read.active
            excel_rows = ws_read.max_row - 1
            print(f"salary.xlsx created and read back: {excel_rows} rows")

def read_pdf_preview(pdf_path: Path) -> None:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    text = page.get_text()
    doc.close()
    print(text[:100])

def read_file(path: Union[str, Path]) -> Optional[dict]:
    path_obj = Path(path)
    try:
        with path_obj.open(mode='r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {path_obj.name}")
        return None
    except json.JSONDecodeError:
        logger.error(f"JSON decode failed: {path_obj.name}")
        return None

def find_employee(name: str, json_path: Path) -> Optional[dict]:
    try:
        with json_path.open(mode='r', encoding='utf-8') as f:
            data: List[dict] = json.load(f)
            for emp in data:
                if emp['name'].lower() == name.lower():
                    return emp
    except Exception:
        pass
    return None

# --- Main Driver Execution Logic ---
if __name__ == "__main__":
    setup_dummy_files()
    
    csv_file = Path("employees.csv")
    json_file = Path("employees.json")
    high_earners_file = Path("high_earners.csv")
    excel_file = Path("salary.xlsx")
    pdf_file = Path("sample.pdf")

    process_employee_data(csv_file, json_file, high_earners_file)
    manage_excel(excel_file, csv_file)
    read_pdf_preview(pdf_file)
    
    # Triggering logging handler exception intentionally for missing.txt
    read_file("missing.txt")
