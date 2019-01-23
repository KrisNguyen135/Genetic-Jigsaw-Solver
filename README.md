# Solving Jigsaw Puzzles with a Genetic Algorithm

<p align="center">
<img src="https://github.com/KrisNguyen135/Genetic-Jigsaw-Solver/blob/master/output/combined.gif" width="600" height="600"/>
</p>

## General information
Jigsaw puzzles are fun! But if you'd like to take the fun out of solving them, it is possible to have computer programs
to solve them for you.

While representing images as matrices of pixels, each piece of a jigsaw puzzle is reduced to a 3D matrix. Furthermore,
it can be quite easy to find out whether two given pieces (or matrices) can be matched together or not, computationally.

### Heuristics and puzzle specifications
For the purpose of this project, I'm only considering square pictures cut into square pieces. This means the solver will
have knowledge regarding the size of the final solution (in terms of how many pieces there are across an edge). This
also means that the method of matching the cut along the edges of two pieces is not viable, since the edges of all the
pieces are cut in a straight line.

### Edge-matching
<p align="center">
<img src="https://image.slidesharecdn.com/gisconcepts3-091126070346-phpapp01/95/gis-concepts-35-61-728.jpg" width="500" height="400"/>
</p> 

<p align="center">
<a href="https://www.slideshare.net/cier/gis-concepts-35">Image source</a>
</p>

Specifically, to see how well two given pieces of the puzzle match horizontally (as indicated in the image above), we
can calculate the difference between the last column of the matrix that represents the piece on the left, and the first
column of the matrix representing the piece on the right.

A small difference between the two columns (or vectors) means
a "smooth" transition from one column of pixels to another. And if the two pieces are indeed next to each other in the
final solution of the jigsaw puzzle, they should have a such smooth transition. The goal is then to match pieces that
have relatively similar edges together.

### The case for the Genetic Algorithm
Say we were to implement a simply brute-force approach to solving a jigsaw puzzle. This means generating all possible
assignments of the individual pieces.

If the original image is cut into ![equation](https://latex.codecogs.com/gif.latex?n) segments along each edge, we will
have ![equation](https://latex.codecogs.com/gif.latex?n%5E2) individual pieces in total, which will lead to
![equation](https://latex.codecogs.com/gif.latex?%28n%5E2%29%21) possible assignments. Furthermore, in a given
assignment, each piece can be placed in exactly four different ways, multiplying the number of potential solutions by
![equation](https://latex.codecogs.com/gif.latex?4%5E%7B%28n%5E2%29%7D).

Let's call the function of the total number of possible assignments of a puzzle cut into
![equation](https://latex.codecogs.com/gif.latex?n) segments along each edge
![equation](https://latex.codecogs.com/gif.latex?f%28n%29). We then have:

![equation](https://latex.codecogs.com/gif.latex?f%28n%29%20%3D%20%28n%5E2%29%21%20*%204%5E%7Bn%5E2%7D)

With a simple 2-by-2 puzzle, we have: ![equation](https://latex.codecogs.com/gif.latex?f%282%29%20%3D%206%2C144)

Increasing the number of segments by one, we have:
![equation](https://latex.codecogs.com/gif.latex?f%283%29%20%3D%2095%2C126%2C814%2C720)

Clearly a simply brute-force approach is not viable.

With a genetic algorithm, we can selectively find good matches among a population of randomly generated assignments at
the beginning. Then these good matches will be preserved and combined with newly-found good matches in future
populations. The complete solution is found when there are enough good matches.

## Future improvements

- More thorough analysis on the optimal way to calculate threshold for good matches
- Implementing concurrency to improve execution time
- Comparing between using grayscale and not using grayscale

## References

[1] Foxworthy, Tyler. _Solving Jigsaw Puzzles with Genetic Algorithms_. [www.youtube.com/watch?v=6DohBytdf6I](https://www.youtube.com/watch?v=6DohBytdf6I)

[2] Sholomon, Dror, Omid E. David, and Nathan S. Netanyahu. "An automatic solver for very large jigsaw puzzles using
genetic algorithms." Genetic Programming and Evolvable Machines 17.3 (2016): 291-313.