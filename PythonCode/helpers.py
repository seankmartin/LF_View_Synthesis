def print_config(config):
    print('Config file contains: ')
    y = [{x, tuple(config.items(x))} for x in config.sections()]
    for x in y:
        print(x)
