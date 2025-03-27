import turtle
import random

def draw_flower():
    screen = turtle.Screen()
    screen.bgcolor("lightblue")
    screen.title("Colorful Flower")

    flower = turtle.Turtle()
    flower.speed(0)
    flower.hideturtle()

    colors = ["red", "orange", "yellow", "pink", "purple", "blue"]

    # Draw flower petals
    for _ in range(36):
        flower.color(random.choice(colors))
        flower.begin_fill()
        flower.circle(100, 60)
        flower.left(120)
        flower.circle(100, 60)
        flower.left(120)
        flower.end_fill()
        flower.right(10)

    # Draw flower center
    flower.penup()
    flower.goto(0, 0)
    flower.pendown()
    flower.color("yellow")
    flower.begin_fill()
    flower.circle(30)
    flower.end_fill()

    # Display message
    flower.penup()
    flower.goto(0, -180)
    flower.color("darkgreen")
    flower.write("You are doing very well,\nkeep it up my favourite girl!", 
                 align="center", font=("Arial", 16, "bold"))

    screen.mainloop()

draw_flower()
