import turtle
import random

def draw_petal():
    turtle.circle(100, 60)
    turtle.left(120)
    turtle.circle(100, 60)
    turtle.left(120)

def draw_flower():
    turtle.speed(0)
    turtle.hideturtle()
    
    # Draw petals
    colors = ["red", "pink", "purple", "orange", "yellow"]
    for _ in range(6):
        turtle.color(random.choice(colors))
        turtle.begin_fill()
        draw_petal()
        turtle.end_fill()
        turtle.left(60)
    
    # Draw center
    turtle.color("yellow")
    turtle.penup()
    turtle.goto(0, 0)
    turtle.pendown()
    turtle.begin_fill()
    turtle.circle(20)
    turtle.end_fill()
    
    # Draw stem
    turtle.color("green")
    turtle.penup()
    turtle.goto(0, -200)
    turtle.pendown()
    turtle.goto(0, 0)

    # Display message
    turtle.penup()
    turtle.goto(0, -250)
    turtle.color("black")
    turtle.write("You are doing very well, keep it up my favourite girl!", align="center", font=("Arial", 16, "bold"))

    turtle.done()

draw_flower()
