import os
from PIL import Image, ImageDraw
import numpy as np
import argparse
import random
import math
import time
import cv2
import ffmpeg


########################################################################################################################
# MAZE GENERATION AND SOLVING
########################################################################################################################

def init_maze(width, height):
    return np.full((width - 1) * height, True), np.full((height - 1) * width, True)


def build_maze(width, height, v_walls, h_walls, record_action=None):
    ids = np.arange(width * height)
    queue = [(True, i) for i in range((width - 1) * height)] + [(False, i) for i in range((height - 1) * width)]
    while len(queue) > 0:
        choice = random.randint(0, len(queue) - 1)
        axis, index = queue[choice]
        if axis:
            i = index // (width - 1)
            j = index % (width - 1)
            old_id = ids[i * width + j + 1]
            new_id = ids[i * width + j]

            if v_walls[index] and old_id != new_id:
                v_walls[index] = False
                ids[ids == old_id] = new_id

                # Record actions
                if record_action is not None:
                    record_action.append(f"select v {index}")
                    record_action.append(f"break v {index} {old_id} {new_id}")
        else:
            old_id = ids[index + width]
            new_id = ids[index]

            if h_walls[index] and old_id != new_id:
                h_walls[index] = False
                ids[ids == old_id] = new_id
                # Record actions
                if record_action is not None:
                    record_action.append(f"select h {index}")
                    record_action.append(f"break h {index} {old_id} {new_id}")

        del queue[choice]

    # Record last ids (cheap trick to know on which id to put the default bg color on)
    if record_action is not None:
        record_action.append(f"end_id {ids[0]}")


def get_neighbors(width, height, v_walls, h_walls, index):
    # find neighbors
    i = index // width
    j = index % width
    n = []  # neighbors
    # East
    if 0 <= j < width - 1 and not v_walls[i * (width - 1) + j]:
        n.append(index + 1)
    # West
    if 0 < j <= width - 1 and not v_walls[i * (width - 1) + j - 1]:
        n.append(index - 1)
    # North
    if 0 < i <= height - 1 and not h_walls[index - width]:
        n.append(index - width)
    # South
    if 0 <= i < height - 1 and not h_walls[index]:
        n.append(index + width)
    return n


def solve_dijkstra(width, height, v_walls, h_walls, start, end=None, record_action=None):
    weights = np.full(width * height, width * height - 1)
    weights[start] = 0
    queue = [start]
    while len(queue) > 0:
        # find next min cell
        index = min(queue, key=lambda k: weights[k])

        # Record select action
        if record_action is not None:
            record_action.append(f"select {index}")

        # Stop search if found or compute all weights
        if end is not None and index == end:
            break
        queue.remove(index)
        n = get_neighbors(width, height, v_walls, h_walls, index)
        n_choices = [i for i in n if weights[index] < weights[i]]
        for c in n_choices:
            # update neighbors
            weights[c] = weights[index] + 1  # path weight is 1
            # push to queue
            if c not in queue:
                queue.append(c)

        if record_action is not None and len(n_choices) > 0:
            record_action.append("update" + "".join([f" {i}" for i in n_choices]))

    return weights


def backtrace_solution(width, height, v_walls, h_walls, start, end, weights):
    index = end
    path = [index]
    while index != start:
        n = get_neighbors(width, height, v_walls, h_walls, index)
        index = min(n, key=lambda k: weights[k])
        path.append(index)
    return path


# Choose a path that is of length x% of the longest path possible
def generate_end_percent(start, weights, percent):
    max_length = np.max(weights)
    end = start
    while end == start:
        choices = np.nonzero(weights >= math.ceil(max_length * percent))[0]
        end = np.random.choice(choices)
    return end


def random_start(width, height):
    return random.randint(0, width * height - 1)


########################################################################################################################
# MAZE PRINTING
########################################################################################################################
def print_closed_h_walls(width):
    for j in range(width):
        print("+---", end="")
    print("+")


def print_h_walls(width, line, h_walls):
    for j in range(width):
        if h_walls[width * line + j]:
            print("+---", end="")
        else:
            print("+   ", end="")
    print("+")


def print_v_walls(width, line, v_walls, weights=None, path=None):
    start, end = -1, -1
    if path is not None:
        start, end = path[0], path[-1]

    print("|", end="")
    for j in range(width - 1):
        # Cell Data
        cell_index = line * width + j

        # Weight
        if weights is not None:
            p = str(weights[cell_index])
            white = " " * (3 - len(p))
            filler = p + white
        else:
            filler = "   "

        # Solution
        if path is not None and cell_index in path:
            filler = " . "

        # Start/end
        if cell_index == start:
            filler = " S "
        elif cell_index == end:
            filler = " E "

        # Wall
        if v_walls[(width - 1) * line + j]:
            print(filler + "|", end="")
        else:
            print(filler + " ", end="")

    # Cell Data
    cell_index = line * width + width - 1

    # Weight
    if weights is not None:
        p = str(weights[cell_index])
        white = " " * (3 - len(p))
        filler = p + white
    else:
        filler = "   "

    # Solution
    if path is not None and cell_index in path:
        filler = " . "

    # Start/end
    if cell_index == start:
        filler = " S "
    elif cell_index == end:
        filler = " E "

    # Wall
    print(filler + "|")


def print_maze(width, height, v_walls, h_walls, weights=None, path=None):
    print_closed_h_walls(width)
    for i in range(height - 1):
        print_v_walls(width, i, v_walls, weights, path)
        print_h_walls(width, i, h_walls)
    print_v_walls(width, height - 1, v_walls, weights, path)
    print_closed_h_walls(width)


########################################################################################################################
# MAZE DRAWING
########################################################################################################################
def draw_v_walls(width, v_walls, line, imgd, size, line_width=1, selected=None):
    for j in range(width - 1):
        if v_walls[(width - 1) * line + j]:
            color = (255, 0, 255) if selected is not None and selected == (width - 1) * line + j else (255, 255, 255)
            imgd.line([((j + 1) * size, line * size), ((j + 1) * size, (line + 1) * size)], fill=color,
                      width=line_width)


def draw_h_walls(width, h_walls, line, imgd, size, line_width=1, selected=None):
    for j in range(width):
        if h_walls[width * line + j]:
            color = (255, 0, 255) if selected is not None and selected == width * line + j else (255, 255, 255)
            imgd.line([(j * size, (line + 1) * size), ((j + 1) * size, (line + 1) * size)], fill=color,
                      width=line_width)


def draw_mouse(width, start, img, size):
    mouse_pos = ((start % width) * size, (start // width) * size)
    # add mouse
    mouse = Image.open("res/mouse.png")
    mouse.thumbnail((size - 2, size - 2), Image.LANCZOS)
    img.paste(mouse, mouse_pos, mouse)


def draw_cheese(width, end, img, size):
    # add cheese
    cheese_pos = ((end % width) * size, (end // width) * size)
    cheese = Image.open("res/cheese.png")
    cheese.thumbnail((size - 2, size - 2), Image.LANCZOS)
    img.paste(cheese, cheese_pos, cheese)


def draw_solution(width, path, imgd, size, line_width=1, color=(255, 0, 0)):
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]
        x1, y1 = start % width, start // width
        x2, y2 = end % width, end // width
        imgd.line([x1 * size + size / 2, y1 * size + size / 2, x2 * size + size / 2, y2 * size + size / 2],
                  fill=color, width=line_width)


# selected = (IS_VERTICAL_SELECTION, INDEX), ex: (True, 0)
def draw_walls(width, height, v_walls, h_walls, imgd, size, line_width=1, selected=None):
    v_select = selected[1] if selected is not None and selected[0] else None
    h_select = selected[1] if selected is not None and not selected[0] else None

    imgd.rectangle([(0, 0), (width * size - 1, height * size - 1)], fill=None, outline=(255, 255, 255),
                   width=line_width)
    for i in range(height - 1):
        draw_v_walls(width, v_walls, i, imgd, size, line_width, v_select)
        draw_h_walls(width, h_walls, i, imgd, size, line_width, h_select)
    draw_v_walls(width, v_walls, height - 1, imgd, size, line_width, v_select)


def draw_ids(width, ids, colors, imgd, size):
    for i in range(len(weights)):
        x, y = i % width, i // width
        imgd.rectangle([x * size, y * size, x * size + size, y * size + size], fill=colors[ids[i]], outline=None)


def heat_rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def draw_heatmap(width, weights, imgd, size, weight_overrides=None):
    m = np.max(weights)
    for i in range(len(weights)):
        if (weight_overrides is not None and weight_overrides[i]) or (weight_overrides is None):
            x, y = i % width, i // width
            color = heat_rgb(0, m, weights[i])
            imgd.rectangle([x * size, y * size, x * size + size, y * size + size], fill=color, outline=None)


def draw_maze(width, height, v_walls, h_walls, start, end, img, imgd, size, line_width=1, path=None, weights=None,
              weight_overrides=None):
    # Draw heatmap
    if weights is not None:
        draw_heatmap(width, weights, imgd, size, weight_overrides)
    # Draw walls
    draw_walls(width, height, v_walls, h_walls, imgd, size, line_width)
    # Draw solution
    if path is not None:
        draw_solution(width, path, imgd, size, line_width, color=(255, 0, 255))
    # Draw start and end positions
    draw_mouse(width, start, img, size)
    draw_cheese(width, end, img, size)


def create_maze_image(width, height, v_walls, h_walls, start, end, size, bg_color=(0, 0, 0), line_width=1, path=None,
                      weights=None, weight_overrides=None):
    img = Image.new(mode="RGB", size=(size * width, size * height), color=bg_color)
    img1 = ImageDraw.Draw(img)
    draw_maze(width, height, v_walls, h_walls, start, end, img, img1, size, line_width, path=path,
              weights=weights, weight_overrides=weight_overrides)
    return img


########################################################################################################################
# MAZE Animation
########################################################################################################################

# Generation
def animate_generation(video, width, height, size, recorded_actions, bg_color=(0, 0, 0), line_width=1):
    ids = np.arange(width * height)
    colors = [(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)) for _ in ids]
    colors[int(recorded_actions[-1].split(" ")[1])] = bg_color  # get the final ids to assign bg color to
    v_walls, h_walls = np.full((width - 1) * height, True), np.full((height - 1) * width, True)
    for a in recorded_actions:
        command = a.split(" ")
        if command[0] == "end_id":
            break
        img = Image.new(mode="RGB", size=(size * width, size * height), color=bg_color)
        img1 = ImageDraw.Draw(img)
        if command[0] == "select":
            axis, index = command[1] == "v", int(command[2])
            # Draw maze
            draw_ids(width, ids, colors, img1, size)
            draw_walls(width, height, v_walls, h_walls, img1, size, line_width=line_width, selected=(axis, index))
        elif command[0] == "break":
            axis, index, old_id, new_id = command[1] == "v", int(command[2]), int(command[3]), int(command[4])
            # break wall
            if axis:
                v_walls[index] = False
            else:
                h_walls[index] = False
            # replace ids
            ids[ids == old_id] = new_id

            # Draw maze
            draw_ids(width, ids, colors, img1, size)
            draw_walls(width, height, v_walls, h_walls, img1, size, line_width=line_width)
        else:
            raise ValueError(f"Unknown action: {command[0]}")

        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))


# weights
def animate_weights(video, width, height, v_walls, h_walls, weights, recorded_actions,
                    size, bg_color=(0, 0, 0), line_width=1):
    weight_overrides = np.full(width * height, False)
    for a in recorded_actions:
        command = a.split(" ")

        if command[0] == "select":
            img = Image.new(mode="RGB", size=(size * width, size * height), color=bg_color)
            img1 = ImageDraw.Draw(img)

            index = int(command[1])
            weight_overrides[index] = True
            # Draw Maze
            draw_heatmap(width, weights, img1, size, weight_overrides=weight_overrides)
            # Selection
            x, y = index % width, index // width
            img1.rectangle((x * size, y * size, x * size + size, y * size + size), fill=(255, 0, 255), outline=None)
            draw_walls(width, height, v_walls, h_walls, img1, size, line_width=line_width)
            draw_mouse(width, start, img, size)
            video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        elif command[0] == "update":
            indexes = [int(i) for i in command[1:]]
            for i in indexes:
                weight_overrides[i] = True


def animate_solve(video, width, height, v_walls, h_walls, weights, path, size,
                  bg_color=(0, 0, 0), line_width=1):
    for i in range(1, len(path)):

        img = Image.new(mode="RGB", size=(size * width, size * height), color=bg_color)
        img1 = ImageDraw.Draw(img)

        # draw heatmap
        draw_heatmap(width, weights, img1, size)
        # draw walls
        draw_walls(width, height, v_walls, h_walls, img1, size, line_width=line_width)

        # draw lines
        for j in range(i):
            start, end = path[j], path[j + 1]
            x1, y1 = start % width, start // width
            x2, y2 = end % width, end // width

            img1.line([x1 * size + size / 2, y1 * size + size / 2, x2 * size + size / 2, y2 * size + size / 2],
                      fill=(255, 0, 255), width=line_width)

        # draw mouse and cheese
        draw_mouse(width, path[-1], img, size)
        draw_cheese(width, path[0], img, size)

        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

########################################################################################################################
# MAIN PROGRAM
########################################################################################################################

if __name__ == "__main__":
    # CLI parser
    parser = argparse.ArgumentParser(prog='maze', description="Maze generator.")

    # Maze generation
    parser.add_argument('-x', '--width', metavar='NB_COLUMNS', dest='w', type=int, required=False, default=10,
                        help='Maze width (default: 10)')
    parser.add_argument('-y', '--height', metavar='NB_LINES', dest='h', type=int, required=False, default=10,
                        help='Maze height (default: 10)')

    parser.add_argument('-s', '--seed', metavar='SEED', dest='seed', type=int, required=False, default=None,
                        help='Use the specified seed')

    parser.add_argument('-t', '--timeit', dest='timeit', required=False, default=False, action='store_true',
                        help="Time maze generation, solving and rendering")

    # Maze printing
    parser.add_argument('-p', '--print', dest='print', required=False, default=False, action='store_true',
                        help="Print the generated maze (walls only)")

    # Maze drawing
    parser.add_argument('-o', '--output', metavar='PATH', dest='output', type=str, required=False, default="",
                        help="Export maze as an image to specified path\nSolved maze is exported to "
                             "'PATH.IMAGE_NAME.solution.FORMAT'")

    parser.add_argument('-c', '--cellsize', metavar='CELL_SIZE', dest='size', type=int, required=False, default=64,
                        help='Cell size in pixels (default: 32)')

    parser.add_argument('-g', '--gradient', dest='gradient', required=False, default=False, action='store_true',
                        help='Draw path length heatmap')

    # Maze animation
    parser.add_argument('-v', '--video', nargs='?', metavar='FPS', dest='video',  type=int, required=False, const=10,
                        help="Export maze generation and solving as an mp4 video with the specified FPS (default: 10). '-o' REQUIRED")

    args = parser.parse_args()

    # random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Maze size
    width, height = args.w, args.h

    # Build and solve maze
    start_build_time = time.process_time()
    recorded_gen_actions = [] if args.video is not None else None
    v_walls, h_walls = init_maze(width, height)
    build_maze(width, height, v_walls, h_walls, record_action=recorded_gen_actions)
    end_build_time = time.process_time()

    # Timeit generation
    if args.timeit:
        print(f"Generation took: {end_build_time - start_build_time} s")

    # Start solution
    start_solve_time = time.process_time()
    start = random_start(width, height)
    # Weights are needed to generate random end
    recorded_weights_actions = [] if args.video is not None else None
    weights = solve_dijkstra(width, height, v_walls, h_walls, start, record_action=recorded_weights_actions)
    end = generate_end_percent(start, weights, 0.75)  # 0.75 is a nice spot
    # Generate the shortest path
    path = backtrace_solution(width, height, v_walls, h_walls, start, end, weights)
    end_solve_time = time.process_time()

    # Timeit solving
    if args.timeit:
        print(f"Solving took: {end_solve_time - start_solve_time} s")

    # Wall and path display width for image export
    line_width = 2 * (args.size // 32)
    if line_width < 1:
        line_width = 1

    # Export as an image
    if args.output != "":
        start_rendering_time = time.process_time()
        # Create images
        maze = create_maze_image(width, height, v_walls, h_walls, start, end, args.size, line_width=line_width,
                                 path=None,
                                 weights=weights if args.gradient else None)

        solved_maze = create_maze_image(width, height, v_walls, h_walls, start, end, args.size, line_width=line_width,
                                        path=path,
                                        weights=weights if args.gradient else None)

        # Save images to disk
        base_path = "".join(args.output.split(".")[:-1])
        maze.save(base_path+".solution."+args.output.split(".")[-1])
        solved_maze.save(args.output)

        end_rendering_time = time.process_time()

        # Timeit render
        if args.timeit:
            print(f"Rendering took: {end_rendering_time - start_rendering_time} s")

    # Print the maze
    if args.print:
        print_maze(width, height, v_walls, h_walls)

    # Generate animation
    if args.video is not None and args.output != "":
        base_path = "".join(args.output.split(".")[:-1])
        tmp_name = f"{base_path}.tmp.mp4"
        video_name = f"{base_path}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(tmp_name, fourcc , args.video, (width * args.size, height * args.size))
        animate_generation(video, width, height, args.size, recorded_gen_actions,
                                   line_width=line_width)
        animate_weights(video, width, height, v_walls, h_walls, weights,
                                recorded_weights_actions, args.size, line_width=line_width)
        animate_solve(video, width, height, v_walls, h_walls, weights, path, args.size,
                              line_width=line_width)
        solved_maze = create_maze_image(width, height, v_walls, h_walls, start, end, args.size, line_width=line_width,
                                        path=path)
        video.write(cv2.cvtColor(np.array(solved_maze), cv2.COLOR_RGB2BGR))
        video.release()

        # compress video
        if os.path.exists(video_name):
            os.remove(video_name)

        stream = ffmpeg.input(tmp_name)
        stream = ffmpeg.output(stream, video_name, loglevel="quiet")
        ffmpeg.run(stream)

        # clear tmp video file
        os.remove(tmp_name)

