#!/usr/bin/gnuplot

set terminal qt persist
set title "p-Laplacian solution with FV discretization"
set xlabel "x"
set ylabel "u_p(x)"

set grid xtics ytics mxtics mytics
plot for [p in "2.0 2.1 2.5 3.1 4.1 5.1"] 'solution_p_'.p.'.dat' u 1:2 w steps t 'p = '.p
