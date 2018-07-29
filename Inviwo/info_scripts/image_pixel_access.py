#Inviwo Python script 
import inviwopy


app = inviwopy.app
network = app.network

print(network.canvases[0].__dir__())
print(network.canvases[0].image.__dir__())
print(network.canvases[0].image.depth.__dir__())
print(network.canvases[0].image.depth.data.__dir__())
print(network.canvases[0].image.depth.data.dtype)
print(network.canvases[0].image.depth.data[0][0])