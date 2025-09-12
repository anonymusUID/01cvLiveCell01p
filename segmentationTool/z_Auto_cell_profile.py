
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
    
    global file_name, image_path
    
    out_dir="Output/Results/"


    
    if len(sys.argv) >3:
       print("Usage: python handshaking error:", sys.argv[1])
       sys.exit(1)
    image_path = sys.argv[1]
    neighborhood_size=sys.argv[2]
    print(image_path)

    
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))  
       
    input_file= out_dir+f"0_{file_name}_{neighborhood_size}"+".xlsx"
    out_prefix= out_dir+f"0_{file_name}_{neighborhood_size}"
       
    jr.generate_report_from_excel(input_file, out_prefix+"_report.html")
    
    

if __name__ == "__main__":
    main()
    
