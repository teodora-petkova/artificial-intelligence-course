import turtle
#import random

def draw_branch(branch_length, angle):
    if branch_length > 5:
        turtle.forward(branch_length)
        turtle.right(angle)
        draw_branch(branch_length - 15, angle)
        turtle.left(2 * angle)
        draw_branch(branch_length - 15, angle)
        turtle.right(angle)
        turtle.backward(branch_length)
#turtle.speed("slowest")
#draw_branch(40, 20)

def draw_tree(trunk_length, angle):
    turtle.speed("slowest")
    turtle.left(90)
    turtle.up()
    turtle.backward(trunk_length)
    turtle.down()
    draw_branch(trunk_length, angle)
    turtle.done()

#draw_tree(100, 10)
#draw_tree(100, 20)
#draw_tree(100, 30)
#draw_tree(100, 45)
#draw_tree(100, 90)
#turtle.done()

class Grammar:
  def __init__(self, rules, axiom, angle):
    self.rules = rules
    self.axiom = axiom
    self.angle = angle

def normalise(value):
    if value > 1:
        return value - 1
    return value

def change_color(color):
    #new_r = random.uniform(0, 1)
    #new_b = random.uniform(0, 1)
    #new_g = random.uniform(0, 1)    
    new_r = normalise(color[0] + 0.002)
    new_b = normalise(color[1] + 0.001)
    new_g = normalise(color[2] + 0.002)
    turtle.pencolor(new_r, new_b, new_g)

def draw_L_system(grammar, generation_count):
    geneneration = get_generation(grammar, generation_count)
    for r in geneneration:
        if r == 'F':
            turtle.forward(5)
        if r == '+':
            turtle.right(grammar.angle)
        if r == '-':
            turtle.left(grammar.angle)
            change_color(turtle.pencolor())
    turtle.done()

def get_generation(grammar, generation_count):
    current_generation = None
    i = 0
    while(i<generation_count):
        i+=1
        current_generation = apply_grammar(grammar, current_generation)
    return current_generation

def apply_grammar(grammar, current_generation):
    if current_generation is None: 
        current_generation = grammar.axiom
    new_generation = ""
    for s in current_generation:
        new_generation += grammar.rules.get(s, s)
    return new_generation

von_koch_snowflakes_grammar = Grammar(
    rules = {
                "F" : "F-F++F-F"
            },
    axiom = "F++F++F",
    angle = 60
)

dragon_curve_grammar = Grammar(
    rules = {
                "X" : "X+YF+",
                "Y" : "-FX-Y"
            },
    axiom = "FX", 
    angle = 90 
)

turtle.speed("fastest")
turtle.hideturtle()
turtle.colormode(1)
turtle.pencolor((0.2, 0.8, 0.55))

#draw_L_system(von_koch_snowflakes_grammar, 11)
draw_L_system(dragon_curve_grammar, 7)