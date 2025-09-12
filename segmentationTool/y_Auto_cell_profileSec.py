
#from jitprofiler import report_ge_v44
#import jitsreport as jr
#pip install -i https://test.pypi.org/simple/ jitprofiler==0.2.0

import jitprofiler as jr


#from jitprofiler import generate_report_from_excel
#jr.generate_report_from_excel("out_34.xlsx","test.html")

# Generate a report
# jr.generate_report_from_excel("data_bri.xlsx", "out.html")


import sys
import os

def main():
    # 1. Check if the correct number of arguments (2 paths) is provided.
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <input_excel_path> <output_html_path>")
        sys.exit(1)

    # 2. Get the full paths directly from the command-line arguments
    input_excel_path = sys.argv[1]
    output_html_path = sys.argv[2]

    print(f"Input Excel file: {input_excel_path}")

    # 3. Check if the input file actually exists before processing
    if not os.path.exists(input_excel_path):
        print(f"❌ Failed to read Excel file: [Errno 2] No such file or directory: '{input_excel_path}'")
        sys.exit(1)

    try:
        # 4. Call the report generator with the correct, full paths
        jr.generate_report_from_excel(input_excel_path, output_html_path)
        #print(f"✅ Report saved to {output_html_path}")
        
    except Exception as e:
        print(f"❌ An error occurred while generating the report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    

