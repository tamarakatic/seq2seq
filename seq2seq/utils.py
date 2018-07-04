COLORS = dict(
    green='32m',
    red='31m',
    blue='34m',
    white='39m'
)


def color(text, color):
    color_code = COLORS.get(color, '39m')
    return '\033[1;{color}{text}\033[0m'.format(text=text, color=color_code)
