# Solving Jigsaw Puzzles with a Genetic Algorithm

## Introduction
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
<p align="center", style="font-size:50%;">
<img src="https://image.slidesharecdn.com/gisconcepts3-091126070346-phpapp01/95/gis-concepts-35-61-728.jpg" width="500" height="400"/>
</p> 

<p align="center", style="font-size:50%;">
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

If the original image is cut into $n$ ![equation](https://latex.codecogs.com/gif.latex?n) segments along each edge, we
will have $n^2$ individual pieces in total, which will lead to $(n^2)!$ possible assignments. Furthermore, in a given
assignment, each piece can be placed in exactly four different ways, multiplying the number of potential solutions by
$4^{n^2}$.

Let's call the function of the total number of possible assignments of a puzzle cut into $n$ segments along each edge
$f(n)$. We then have:

$$f(n) = (n^2)! * 4^{n^2}$$

With a simple 2-by-2 puzzle, we have: $f(2) = 6,144$

Increasing the number of segment by one, we have: $f(3) = 95,126,814,720$

Clearly a simply brute-force approach is not viable.

## Methodologies

## Future improvements

- More thorough analysis on the optimal way to calculate threshold for good matches
- Optimizing the calculation of fitness (maybe use the cluster matrix during)
- Implementing concurrency to improve execution time
- Comparing between using grayscale and not using grayscale

## References

[1] Foxworthy, Tyler. _Solving Jigsaw Puzzles with Genetic Algorithms_. [www.youtube.com/watch?v=6DohBytdf6I](https://www.youtube.com/watch?v=6DohBytdf6I)
