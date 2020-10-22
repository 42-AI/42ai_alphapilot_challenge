from PIL import Image, ImageDraw
from random import randint


def fake_light(image, tilesize=20, color=(255, 255, 255), alpha=128):
    width, height = image.size
    r, g, b = color
    for x in range(0, width, tilesize):
        for y in range(0, height, tilesize):
            coef = 1 - x / float(width) * y / float(height)
            new_color = int(r * coef), int(g * coef), int(b * coef)
            tile = Image.new("RGBA", (tilesize, tilesize), (255, 255, 255, alpha))
            image.paste(new_color, (x, y, x + tilesize, y + tilesize), mask=tile)


def light_ellipse(img, alpha=False, color=True):
    def colored(i, color, delta):
        color = [255 - c * i for c in color]
        color = tuple([c if c >= 0 else 0 for c in color]) + (delta,)
        return color

    delta = randint(5, 20) if alpha else 5
    tile = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(tile)
    width, height = img.size
    x, y = randint(0, width), randint(0, height)
    dir = [randint(2, 30) for _ in range(4)]
    r, g, b = [randint(1, 10) for _ in range(3)] if color else [randint(1, 10)]*3
    i = 1
    while i < 30:
        n, s, w, e = [i * val for val in dir]
        west_north = (x - w, y - n)
        east_south = (x + e, y + s)
        fill_color = colored(i, (r, g, b), delta)
        draw.ellipse([west_north, east_south], fill=fill_color)
        i += 1
        img.paste(tile, (0, 0), mask=tile)
    img.putalpha(255)
