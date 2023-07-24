# AnotherMazeGenerator

<img src="https://github.com/smallcluster/AnotherMazeGenerator/blob/images/home.png?raw=true" width="250px" align="left">

Yet again, another CLI maze generator written in **Python3**, but with a cool visualization to video!

Help Mr.Squeaks 🐁 to get that 🧀! But beware of the sticky spider webs 🕷️🕸️, they will slow you down!

## Why ?

Python being the **FASTEST** 🚀 language known to man, you can take the coffee break **YOU** 🫵 deserve and get your self a nice maze with its generation and solving steps in a video when you come back.

---

## 📦 Requirements

Install the required libraries:

```shell
pip install -r requirements.txt
```

- numpy
- pillow
- opencv-python

## > Basic usage

### 📖 Help

To show all CLI options use the `-h` option:

```shell
python maze.py -h
```

### 🖼️ Create a maze

Just specify image name (with its format) to the `-o` argument. Two images of a maze will be generated:

- without a solution
- with a solution (the shortest path)

**Example:**

```shell
python maze.py -o example.png
```

<p align="center">
    <img src="https://github.com/smallcluster/AnotherMazeGenerator/blob/images/default.png?raw=true" width="200px"> <img src="https://github.com/smallcluster/AnotherMazeGenerator/blob/images/default.solution.png?raw=true" width="200px"> <br>
    example.png (left), example.solution.png (right)
</p>

#### 🏋️ Show weights

Maze solving uses the *dijkstra* algorithm to compute all cells weights (min path length from source). 

The `-g` option, will draw those weights in the form of a heatmap.

**Example:**

```shell
python maze.py -o example.png -g
```

<p align="center">
    <img src="https://github.com/smallcluster/AnotherMazeGenerator/blob/images/weights.png?raw=true" width="200px"> <img src="https://github.com/smallcluster/AnotherMazeGenerator/blob/images/weights.solution.png?raw=true" width="200px"> <br>
    example.png (left), example.solution.png (right)
</p>


#### 🕸️ Add webs 

TODO: document this

**Example:**

```shell
python maze.py -o example.png -w "4;0.1" -b 0.2
```

<p align="center">
    <img src="https://github.com/smallcluster/AnotherMazeGenerator/blob/images/webs.png?raw=true" width="200px"> <img src="https://github.com/smallcluster/AnotherMazeGenerator/blob/images/webs.solution.png?raw=true" width="200px"> <br>
    example.png (left), example.solution.png (right)
</p>

### 🎬 Visualize the algorithms

If an image is specified, one can use the `-v` option to generate a mp4 video showing the maze generation and maze solving.

**Example:**

```shell
python maze.py -o example.png -v
```

<p align="center">
<img src="https://github.com/smallcluster/AnotherMazeGenerator/blob/images/example.gif?raw=true" width="320"/>
</p>

⚠️ Compressing the generated video is highly recommended as the default compression isn't aggressive enough (video is mostly static)

**Example with ffmpeg:**

```shell
ffmpeg -i example.mp4 tmp.mp4 # up to 80% reduction with negligible quality loss
rm example.mp4
mv tmp.mp4 example.mp4
```