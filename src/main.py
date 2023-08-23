import random
from random import randint
from threading import Thread

import numpy as np
import pygame
import sys
from pygame.locals import *

colors = {'red': (255, 0, 0),
          'green': (0, 255, 0),
          'blue': (0, 0, 255),
          'white': (255, 255, 255),
          'grey': (128, 128, 128),
          'black': (0, 0, 0),
          'silver': (192, 192, 192),
          'purple': (128, 0, 128),
          'yellow': (255, 255, 0),
          'olive': (128, 128, 0),
          'grass': (0, 128, 0),
          'aqua': (0, 255, 255),
          'navy': (0, 0, 128)}
color_keys = list(colors.keys())
color_values = list(colors.values())
delta_t = 1

# constants
k = 1.38065e-23
R = 8.3145
mass_correction = 1.67e-27
speed_correction = 500
WIDTH = 1280
HEIGHT = 720
FPS = 120
interface_width = 400


def maxwell_distribution(v, mass, T):
    return mass / T * v ** 2 * np.exp(-mass * v ** 2 / (2 * k * T))


pygame.init()


class Sphere:
    def __init__(self, x, y, radius=10, mass=1, direction=randint(0, 360), speed=random.random(),
                 color=random.choice(color_values)):
        self.x = x
        self.y = y
        self.x_next = x
        self.y_next = y
        self.radius = radius
        self.exist_matrix = np.zeros((2 * self.radius - 1, 2 * self.radius - 1), dtype=int)
        self.mass = mass
        self.direction = direction  # in degrees
        self.speed = speed
        self.color = color
        self.index = 1
        self.FPT = 0
        self.FPL = 0
        self.impacts = 0
        self.create_exist_matrix()

    def set_index(self, index):
        self.index = index
        self.recreate_exist_matrix()

    def create_exist_matrix(self):
        for m_x in range(2 * self.radius - 1):
            for m_y in range(2 * self.radius - 1):
                if round((m_x - self.radius + 1) ** 2 + (m_y - self.radius + 1) ** 2) <= (
                        (2 * self.radius - 1) / 2) ** 2:
                    self.exist_matrix[m_y, m_x] = self.index

    def recreate_exist_matrix(self):
        self.exist_matrix *= self.index

    def next_point(self, time_s):
        self.x_next = self.x + self.speed * time_s * np.cos(self.direction * np.pi / 180)
        self.y_next = self.y + self.speed * time_s * np.sin(self.direction * np.pi / 180)

    def move(self, time_s):
        self.x += self.speed * time_s * np.cos(self.direction * np.pi / 180)
        self.y += self.speed * time_s * np.sin(self.direction * np.pi / 180)


class Field(Thread):
    def __init__(self, field_size_x=WIDTH, field_size_y=HEIGHT, color=colors['black']):
        Thread.__init__(self)
        self.field_size_x = field_size_x
        self.field_size_y = field_size_y
        self.wall_width = 10
        self.working_area = (self.field_size_x - 2 * self.wall_width) * (self.field_size_y - 2 * self.wall_width)
        self.rect = pygame.Rect((0, 0, self.field_size_x, self.field_size_y))
        self.color = color
        self.time_rate = 1.0
        self.is_running = False
        self.spheres = np.zeros(999, dtype=Sphere)
        self.adding_queue = list()
        self.radii = np.zeros(999)
        self.FPLs = np.zeros(999)
        self.FPL_integrate = 0
        self.FPL_theory = 0
        self.correction_factor = 1 / 1.06
        self.number_of_spheres = 0
        self.tracked_sph_index = 0
        self.busy_map = np.ones((field_size_y, field_size_x), dtype=int) * 999
        self.busy_map[self.wall_width:-self.wall_width, self.wall_width:-self.wall_width] = 0

    def add_sphere(self, x, y, radius=10, mass=1, direction=randint(0, 360), speed=random.random(),
                   color=random.choice(color_values)):
        self.number_of_spheres += 1
        self.spheres[self.number_of_spheres] = Sphere(x, y, radius, mass, direction, speed, color)
        sph = self.spheres[self.number_of_spheres]
        sph.set_index(self.number_of_spheres)
        while True:
            if sph.x < radius + self.wall_width:
                sph.x = radius + self.wall_width
            if sph.x > self.field_size_x - radius - self.wall_width - 1:
                sph.x = self.field_size_x - radius - self.wall_width - 1
            if sph.y < radius + self.wall_width:
                sph.y = radius + self.wall_width
            if sph.y > self.field_size_y - radius - self.wall_width - 1:
                sph.y = self.field_size_y - radius - self.wall_width - 1
            if not np.logical_and(self.busy_map[sph.y - radius + 1:sph.y + radius, sph.x - radius + 1:sph.x + radius],
                                  sph.exist_matrix).any():
                break
            else:
                sph.x = randint(radius + self.wall_width, self.field_size_x - radius - self.wall_width - 1)
                sph.y = randint(radius + self.wall_width, self.field_size_x - radius - self.wall_width - 1)

        self.busy_map[sph.y - radius + 1:sph.y + radius, sph.x - radius + 1:sph.x + radius] += sph.exist_matrix

    def fill(self, number_of_spheres, order="rand", mass=1, radius=10, basic=1):
        if order == "line":
            for number in range(number_of_spheres):
                self.add_sphere(number * 20, number * 20, radius, mass, randint(0, 359), basic * random.random(),
                                colors['black'])
        else:
            for number in range(number_of_spheres):
                self.add_sphere(randint(radius + self.wall_width, self.field_size_x - radius - self.wall_width - 1),
                                randint(radius + self.wall_width, self.field_size_x - radius - self.wall_width - 1),
                                radius, mass, randint(0, 359), basic * random.random(), colors['black'])

    def clear_field(self):
        self.spheres = {0}
        self.number_of_spheres = 0

    def run(self):
        while pygame.get_init():
            for sph in self.spheres[1:]:
                if sph == 0:
                    break
                self.busy_map[int(round(sph.y) - sph.radius + 1):int(round(sph.y) + sph.radius),
                int(round(sph.x) - sph.radius + 1):int(round(sph.x) + sph.radius)] *= np.logical_not(sph.exist_matrix)
                sph.next_point(delta_t * self.time_rate * self.is_running)

                opponents_matrix = np.logical_and(
                    self.busy_map[int(round(sph.y_next) - sph.radius + 1):int(round(sph.y_next) + sph.radius),
                    int(round(sph.x_next) - sph.radius + 1):int(round(sph.x_next) + sph.radius)], sph.exist_matrix)

                if opponents_matrix.any():

                    opponents_list = np.unique(
                        self.busy_map[int(round(sph.y_next) - sph.radius + 1):int(round(sph.y_next) + sph.radius),
                        int(round(sph.x_next) - sph.radius + 1):int(round(sph.x_next) + sph.radius)] * opponents_matrix)

                    for opp in opponents_list:
                        if opp == 0:
                            continue
                        if opp == 999:
                            if sph.x_next + sph.radius >= self.field_size_x - self.wall_width - 1 and np.sign(
                                    self.time_rate) * np.cos(sph.direction * np.pi / 180) > 0:
                                sph.direction = 180 - sph.direction
                                sph.x = self.field_size_x - sph.radius - self.wall_width - 1
                            if sph.x_next - sph.radius <= self.wall_width and np.sign(self.time_rate) * np.cos(
                                    sph.direction * np.pi / 180) < 0:
                                sph.direction = 180 - sph.direction
                                sph.x = sph.radius + self.wall_width
                            if sph.y_next + sph.radius >= self.field_size_y - self.wall_width - 1 and np.sign(
                                    self.time_rate) * np.sin(sph.direction * np.pi / 180) > 0:
                                sph.direction = 360 - sph.direction
                                sph.y = self.field_size_y - sph.radius - self.wall_width - 1
                            if sph.y_next - sph.radius <= self.wall_width and np.sign(self.time_rate) * np.sin(
                                    sph.direction * np.pi / 180) < 0:
                                sph.direction = 360 - sph.direction
                                sph.y = sph.radius + self.wall_width
                            sph.next_point(delta_t * self.time_rate * self.is_running)
                        if 0 < opp < 999:
                            sph.impacts += 1
                            sph.FPL = (sph.FPL * (sph.impacts - 1) + sph.FPT * sph.speed) / sph.impacts
                            fld.spheres[opp].impacts += 1
                            fld.spheres[opp].FPL = (fld.spheres[opp].FPL * (fld.spheres[opp].impacts - 1) +
                                                    fld.spheres[opp].FPT * fld.spheres[opp].speed) \
                                                   / fld.spheres[opp].impacts
                            sph.FPT = 0
                            fld.spheres[opp].FPT = 0
                            # print(sph.index, opp, sph.FPL, fld.spheres[opp].FPL)
                            theta = np.arctan2(self.spheres[opp].y - sph.y, self.spheres[opp].x - sph.x)
                            m_sph = sph.mass
                            m_opp = self.spheres[opp].mass
                            dir_sph_theta = sph.direction * np.pi / 180 - theta
                            dir_opp_theta = self.spheres[opp].direction * np.pi / 180 - theta

                            tmp_v_sph_x = sph.speed * np.cos(dir_sph_theta)
                            tmp_v_sph_y = sph.speed * np.sin(dir_sph_theta)

                            tmp_v_opp_x = self.spheres[opp].speed * np.cos(dir_opp_theta)
                            tmp_v_opp_y = self.spheres[opp].speed * np.sin(dir_opp_theta)

                            tmp_new_v_sph_x = ((m_sph - m_opp) * tmp_v_sph_x + 2 * m_opp * tmp_v_opp_x) / (
                                    m_sph + m_opp)
                            tmp_new_v_opp_x = (2 * m_sph * tmp_v_sph_x + (m_opp - m_sph) * tmp_v_opp_x) / (
                                    m_sph + m_opp)

                            sph.speed = np.sqrt(tmp_new_v_sph_x ** 2 + tmp_v_sph_y ** 2)
                            self.spheres[opp].speed = np.sqrt(tmp_new_v_opp_x ** 2 + tmp_v_opp_y ** 2)

                            tmp_new_dir_sph_theta = np.arctan2(tmp_v_sph_y, tmp_new_v_sph_x)
                            tmp_new_dir_opp_theta = np.arctan2(tmp_v_opp_y, tmp_new_v_opp_x)

                            sph.direction = (tmp_new_dir_sph_theta + theta) * 180 / np.pi
                            self.spheres[opp].direction = (tmp_new_dir_opp_theta + theta) * 180 / np.pi
                        else:
                            pass

                sph.move(delta_t * self.time_rate * self.is_running)
                sph.FPT += delta_t * self.time_rate * self.is_running
                self.busy_map[int(round(sph.y) - sph.radius + 1):int(round(sph.y) + sph.radius),
                int(round(sph.x) - sph.radius + 1):int(round(sph.x) + sph.radius)] += sph.exist_matrix

            self.FPLs = [s.FPL for s in self.spheres[1:self.number_of_spheres + 1]]
            self.radii = [s.radius for s in self.spheres[1:self.number_of_spheres + 1]]
            self.FPL_integrate = np.sum(self.FPLs) / self.number_of_spheres
            self.FPL_theory = self.working_area / (
                    np.sqrt(2) * 4 * np.mean(self.radii) * self.number_of_spheres)
            # print(self.FPL_integrate, self.FPL_theory)
            if len(self.adding_queue):
                fld.add_sphere(self.adding_queue[-1].x, self.adding_queue[-1].y, self.adding_queue[-1].radius,
                               self.adding_queue[-1].mass, self.adding_queue[-1].direction, self.adding_queue[-1].speed,
                               colors['red'])
                fld.tracked_sph_index = fld.number_of_spheres
                input_field_sph_index.set_content(str(fld.number_of_spheres))
                self.adding_queue.pop()


class Button:
    def __init__(self, surface, name, x=0, y=0, width=WIDTH, height=HEIGHT, is_pressed_time=15, color=(200, 255, 200)):
        self.surf = surface
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect((self.x, self.y, self.width, self.height))
        self.color_off = color
        self.color_on = colors['grass']
        self.color = self.color_off
        self.name = name
        self.font = pygame.font.SysFont('Comic Sans MS', 18, True)
        self.marker_surface = self.font.render(self.name, True, colors['black'])
        self.marker_pos = self.marker_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        self.is_pressed = False
        self.is_pressed_time = is_pressed_time
        self.time_on = 0

    def set_marker(self, marker_surface):
        self.marker_surface = marker_surface
        self.marker_pos = marker_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))

    def on(self):
        self.is_pressed = True
        self.color = self.color_on
        self.time_on = self.is_pressed_time

    def off(self):
        self.is_pressed = False
        self.color = self.color_off

    def toggle(self):
        if self.is_pressed:
            self.off()
        else:
            self.on()

    def draw(self):
        if self.is_pressed:
            self.time_on -= 1
        if not self.time_on:
            self.off()
        if pygame.get_init():
            pygame.draw.rect(self.surf, self.color, (self.x, self.y, self.width, self.height), 2, 10)
            self.surf.blit(self.marker_surface, self.marker_pos)


class IOField:
    def __init__(self, surface, x=0, y=0, width=WIDTH, height=HEIGHT, color=colors['white']):
        self.surf = surface
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect((self.x, self.y, self.width, self.height))
        self.color = color
        self.content = ""
        self.font = pygame.font.SysFont('Comic Sans MS', 30)
        self.content_surface = self.font.render(self.content, True, colors['black'])
        self.content_pos = self.content_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        self.is_entered = False

    def render(self):
        self.content_surface = self.font.render(self.content, True, colors['black'])
        self.content_pos = self.content_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))

    def on(self):
        self.is_entered = True
        self.color = colors['white']

    def off(self):
        self.is_entered = False
        self.color = (210, 210, 210)

    def handler(self, event):
        if self.is_entered:
            if event.key == K_KP_ENTER or event.key == K_RETURN:
                self.off()
            if event.key == K_BACKSPACE:
                self.content = self.content[:-1]
            elif len(self.content) < 3:
                self.add_content(event.unicode)

    def set_content(self, string=""):
        self.content = string

    def add_content(self, char=""):
        if self.is_entered:
            if char.isdigit():
                self.content += char

    def draw(self):
        if pygame.get_init():
            pygame.draw.rect(self.surf, self.color, (self.x, self.y, self.width, self.height))
            self.render()
            self.surf.blit(self.content_surface, self.content_pos)


fld = Field(WIDTH - interface_width, HEIGHT, colors['silver'])
# fld.add_sphere(400, HEIGHT // 2, 30, 5, 0, 0, colors['purple'])
# fld.add_sphere(30, HEIGHT // 2, 30, 1000, 0, 0.5, colors['red'])
fld.fill(200, "", 10, 5, 0.3)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
data_font = pygame.font.SysFont('Comic Sans MS', 20)
pygame.display.set_caption("Visualization window")
clock = pygame.time.Clock()
running = True

button_speed_dec = Button(screen, "SpeedDown", 890, 580, 110, 60, 15)
button_start = Button(screen, "Start/Pause", 1010, 580, 140, 60, np.inf)
button_speed_inc = Button(screen, "SpeedUp", 1160, 580, 110, 60, 15)

input_field_sph_index = IOField(screen, 890, 260, 120, 60, (210, 210, 210))
fld.tracked_sph_index = 1
input_field_sph_index.set_content(str(fld.tracked_sph_index))
button_track_sph = Button(screen, "Track particle", 1020, 260, 250, 60, 15)

# output_field_tracked_sph_header = IOField(screen, 890, 340, 380, 50, (127, 255, 127))
# output_field_tracked_sph_header.set_content("Tracked corpuscle info:")
output_field_tracked_sph_x = IOField(screen, 890, 330, 120, 60, (200, 255, 200))
output_field_tracked_sph_y = IOField(screen, 1020, 330, 120, 60, (200, 255, 200))
output_field_tracked_sph_speed = IOField(screen, 1150, 330, 120, 60, (200, 255, 200))

input_field_sph_x_header = IOField(screen, 890, 420, 25, 60, (200, 255, 200))
input_field_sph_x_header.set_content("x")
input_field_sph_y_header = IOField(screen, 1020, 420, 25, 60, (200, 255, 200))
input_field_sph_y_header.set_content("y")
input_field_sph_speed_header = IOField(screen, 1150, 420, 25, 60, (200, 255, 200))
input_field_sph_speed_header.set_content("v")
input_field_sph_m_header = IOField(screen, 890, 490, 25, 60, (200, 255, 200))
input_field_sph_m_header.set_content("m")
input_field_sph_r_header = IOField(screen, 1020, 490, 25, 60, (200, 255, 200))
input_field_sph_r_header.set_content("r")
input_field_sph_x = IOField(screen, 915, 420, 95, 60, (210, 210, 210))
input_field_sph_y = IOField(screen, 1045, 420, 95, 60, (210, 210, 210))
input_field_sph_speed = IOField(screen, 1175, 420, 95, 60, (210, 210, 210))
input_field_sph_m = IOField(screen, 915, 490, 95, 60, (210, 210, 210))
input_field_sph_r = IOField(screen, 1045, 490, 95, 60, (210, 210, 210))
button_sph_add = Button(screen, "Add", 1150, 490, 120, 60, 15)

output_field_graph = IOField(screen, 890, 10, 380, 240, colors['white'])
output_field_scale_mark = IOField(screen, 890, 650, 250, 60, (255, 215, 0))
output_field_scale_mark.set_content("Time acc.")
output_field_scale_value = IOField(screen, 1150, 650, 120, 60, (210, 210, 210))

fld.start()

while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            # print(event.pos, event.button)
            if button_start.rect.collidepoint(event.pos):
                button_start.toggle()
                fld.is_running = 1 - fld.is_running
            if button_speed_dec.rect.collidepoint(event.pos):
                button_speed_dec.toggle()
                fld.time_rate -= 0.1 if fld.time_rate > -2 else 0
            if button_speed_inc.rect.collidepoint(event.pos):
                button_speed_inc.toggle()
                fld.time_rate += 0.1 if fld.time_rate < 2 else 0
            if input_field_sph_index.rect.collidepoint(event.pos):
                input_field_sph_index.on()
            else:
                input_field_sph_index.off()
            if button_track_sph.rect.collidepoint(event.pos):
                button_track_sph.on()
                tmp = int("0" + input_field_sph_index.content)
                fld.tracked_sph_index = tmp if 0 < tmp <= fld.number_of_spheres else 0
                if not fld.tracked_sph_index:
                    input_field_sph_index.set_content("")
            if input_field_sph_x.rect.collidepoint(event.pos):
                input_field_sph_x.on()
            else:
                input_field_sph_x.off()
            if input_field_sph_y.rect.collidepoint(event.pos):
                input_field_sph_y.on()
            else:
                input_field_sph_y.off()
            if input_field_sph_speed.rect.collidepoint(event.pos):
                input_field_sph_speed.on()
            else:
                input_field_sph_speed.off()
            if input_field_sph_m.rect.collidepoint(event.pos):
                input_field_sph_m.on()
            else:
                input_field_sph_m.off()
            if input_field_sph_r.rect.collidepoint(event.pos):
                input_field_sph_r.on()
            else:
                input_field_sph_r.off()
            if button_sph_add.rect.collidepoint(event.pos):
                button_sph_add.toggle()
                tmp_x = int("0" + input_field_sph_x.content)
                tmp_y = int("0" + input_field_sph_y.content)
                tmp_v = int("0" + input_field_sph_speed.content)
                tmp_r = int("0" + input_field_sph_r.content)
                tmp_m = int("0" + input_field_sph_m.content)
                if tmp_x and tmp_y and tmp_v and tmp_r and tmp_m:
                    fld.adding_queue.append(Sphere(tmp_x, tmp_y, tmp_r, tmp_m, randint(0, 359), tmp_v))
            if fld.rect.collidepoint(event.pos):
                tmp = fld.busy_map[event.pos[1], event.pos[0]]
                fld.tracked_sph_index = tmp if 0 < tmp <= fld.number_of_spheres else 0
                input_field_sph_index.set_content(str(tmp if tmp else ""))

        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
                pygame.quit()
                sys.exit()
            if event.key == K_SPACE:
                button_start.toggle()
                fld.is_running = 1 - fld.is_running
            input_field_sph_index.handler(event)
            input_field_sph_x.handler(event)
            input_field_sph_y.handler(event)
            input_field_sph_speed.handler(event)
            input_field_sph_m.handler(event)
            input_field_sph_r.handler(event)

    screen.fill(colors['red'])
    pygame.draw.rect(screen, colors['white'],
                     (fld.wall_width, fld.wall_width,
                      fld.field_size_x - 2 * fld.wall_width, fld.field_size_y - 2 * fld.wall_width))
    pygame.draw.rect(screen, colors['silver'], (WIDTH - interface_width, 0, WIDTH - 1, HEIGHT))
    # text_surface = data_font.render("TextString", False, (0, 0, 0))
    # screen.blit(text_surface, (890, 650))

    button_start.draw()
    button_speed_dec.draw()
    button_speed_inc.draw()
    input_field_sph_index.draw()
    button_track_sph.draw()

    if fld.tracked_sph_index:
        output_field_tracked_sph_x.set_content(str(round(fld.spheres[fld.tracked_sph_index].x, ndigits=2)))
        output_field_tracked_sph_y.set_content(str(round(fld.spheres[fld.tracked_sph_index].y, ndigits=2)))
        output_field_tracked_sph_speed.set_content(str(round(fld.spheres[fld.tracked_sph_index].speed, ndigits=2)))
    else:
        output_field_tracked_sph_x.set_content("")
        output_field_tracked_sph_y.set_content("")
        output_field_tracked_sph_speed.set_content("")
    # output_field_tracked_sph_header.draw()
    output_field_tracked_sph_x.draw()
    output_field_tracked_sph_y.draw()
    output_field_tracked_sph_speed.draw()

    input_field_sph_x_header.draw()
    input_field_sph_y_header.draw()
    input_field_sph_speed_header.draw()
    input_field_sph_m_header.draw()
    input_field_sph_r_header.draw()
    input_field_sph_x.draw()
    input_field_sph_y.draw()
    input_field_sph_speed.draw()
    input_field_sph_m.draw()
    input_field_sph_r.draw()
    button_sph_add.draw()

    output_field_graph.draw()
    speeds = np.asarray([s.speed for s in fld.spheres[1:fld.number_of_spheres + 1]])
    masses = np.asarray([s.mass for s in fld.spheres[1:fld.number_of_spheres + 1]])
    rms_mass = np.sqrt(sum(masses ** 2) / fld.number_of_spheres)
    rms_speed = np.sqrt(sum(speeds ** 2) / fld.number_of_spheres)
    mean_kinetic = sum(masses * speeds ** 2) / fld.number_of_spheres
    T = 2 * mean_kinetic / (3 * k)
    # print(T)
    # print(speeds)
    # print(rms_speed)
    MLS = np.sqrt(2 * k * T / rms_mass)
    # print(MLS)
    # speeds.sort()
    n_hist, bin_edges = np.histogram(speeds, bins=16, range=(0, MLS * 2), density=False)
    # print(n_hist, bin_edges)
    x_hist_corr = output_field_graph.width / len(n_hist)
    y_hist_corr = output_field_graph.height / np.max(n_hist)
    n_MLS = maxwell_distribution(MLS, rms_mass, T)
    # print(n_MLS)
    y_corr = output_field_graph.height / n_MLS
    # print(y_corr)
    v = np.linspace(0, MLS * 3, 65)
    # print(v)
    x_corr = output_field_graph.width / (MLS * 3)
    # print(x_corr)
    n = maxwell_distribution(v, rms_mass, T)
    # print(n)
    for i in range(len(n_hist)):
        pygame.draw.rect(screen, (0, 0, 255), (890 + i * x_hist_corr + 1, 250 - n_hist[i] * y_hist_corr,
                                               x_hist_corr - 1, n_hist[i] * y_hist_corr))
    for i in range(len(v) - 1):
        pygame.draw.line(screen, (0, 255, 0), (890 + v[i] * x_corr, 250 - n[i] * y_corr),
                         (890 + v[i + 1] * x_corr, 250 - n[i + 1] * y_corr), 2)
    # plt.plot(v,n)
    # plt.show()

    output_field_scale_mark.draw()
    output_field_scale_value.set_content(str(round(fld.time_rate, ndigits=1)))
    output_field_scale_value.color = (127.5 + fld.time_rate * 64, 127, 127.5 - fld.time_rate * 64)
    output_field_scale_value.draw()

    txt = pygame.font.SysFont('Comic Sans MS', 25, True)
    screen.blit(txt.render("n", True, (0, 0, 0)), (895, 5))
    screen.blit(txt.render("v", True, (0, 0, 0)), (1250, 215))
    txt = pygame.font.SysFont('Comic Sans MS', 16, True)
    screen.blit(
        txt.render("<lambda> " + str(round(fld.FPL_integrate)) + "/" + str(round(fld.FPL_theory)), True, (0, 0, 0)),
        (1120, 5))

    for i in range(1, fld.number_of_spheres + 1):
        pygame.draw.circle(screen, fld.spheres[i].color, (fld.spheres[i].x, fld.spheres[i].y), fld.spheres[i].radius)
    if fld.tracked_sph_index:
        pygame.draw.rect(screen, colors['green'],
                         (round(fld.spheres[fld.tracked_sph_index].x) - fld.spheres[fld.tracked_sph_index].radius - 5,
                          round(fld.spheres[fld.tracked_sph_index].y) - fld.spheres[fld.tracked_sph_index].radius - 5,
                          fld.spheres[fld.tracked_sph_index].radius * 2 + 9,
                          fld.spheres[fld.tracked_sph_index].radius * 2 + 9), 1)
    pygame.display.flip()
pygame.quit()
