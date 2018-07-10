#Inviwo Python script 
import inviwopy
import os
import sys

print("Python path for Inviwo is:")
for path in sys.path:
    print(os.path.normpath(path))
print("Please place additional modules in one of the above locations")
print("For example, numpy")
