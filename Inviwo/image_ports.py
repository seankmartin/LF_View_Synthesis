#Inviwo Python script 
import inviwopy


app = inviwopy.app
network = app.network

#print(inviwopy.data.__dir__())
print(inviwopy.data.ImageOutport.__dir__)

print()

print(network.PythonScriptProcessor.__dir__())

help(network.PythonScriptProcessor.getOutport("outport"))

help(network.PythonScriptProcessor.getInport("im_inport1").getData())

#print(network.PythonScriptProcessor.getInport("im_inport1").getData().colorLayers[0].data)

help(network.PythonScriptProcessor.getInport("im_inport1").getData().colorLayers[0].data)