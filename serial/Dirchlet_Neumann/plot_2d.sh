#!/usr/bin/gnuplot

set terminal qt persist
set autoscale xfix 
set autoscale yfix 
set autoscale cbfix
set view map
set palette rgb 33,13,10

set title "p-Laplacian solution with FV discretization"
set xlabel "x(x1)"
set ylabel "y(x2)"

set grid xtics ytics mxtics mytics
set xrange [0:1]
set yrange [0:1]

set multiplot layout 1, 2
set title "Displacements x"
splot "x_solution.dat" nonuniform matrix with linesp ps 3 pt 5 lc palette
set title "Displacements y"
splot "y_solution.dat" nonuniform matrix with linesp ps 3 pt 5 lc palette
unset multiplot
