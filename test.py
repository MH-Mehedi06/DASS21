print("Hello World from Python")
import sys
print(sys.executable)
try:
    import pandas
    print("Pandas imported")
    import openpyxl
    print("Openpyxl imported")
except ImportError as e:
    print(f"Import Error: {e}")
