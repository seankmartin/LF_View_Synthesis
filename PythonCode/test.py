import configparser
import os

#Just to verify things work
config = configparser.ConfigParser()
config.read(os.path.join('config','main.ini'))

print('Program started with the following options')
print('Config', config.sections())
y = [{x, tuple(config.items(x))} for x in config.sections()]
for x in y:
    print(x)
