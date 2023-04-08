import numpy as np
import pyglet

index = 0


def draw_res(res):
    t = res.t
    x = res.y[0, :] * 100 + 100
    y = res.y[1, :] * 100 + 100
    φ = np.degrees(res.y[2, :])
    vx = res.y[3, :]
    vy = res.y[4, :]
    ω = res.y[5, :]

    rec_height = 100
    rec_width = 100

    # Create a window
    window = pyglet.window.Window()
    window.set_fullscreen()
    # Create a rectangle
    rectangle = pyglet.shapes.Rectangle(x=x[0], y=y[0], width=rec_width, height=rec_height)

    @window.event
    def on_draw():
        window.clear()
        rectangle.draw()

    # Move the rectangle to a new position
    def update(dt):
        global index
        if index >= t.size:
            return

        index += 1
        rectangle.x = x[index] + rec_height / 2
        rectangle.y = y[index] + rec_width / 2
        rectangle.rotation = φ[index]

        pyglet.clock.unschedule(update)
        pyglet.clock.schedule_interval(update, t[index] - t[index - 1])

    pyglet.clock.schedule_interval(update, t[0])

    pyglet.app.run()
