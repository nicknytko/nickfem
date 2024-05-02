// Gmsh project created on Fri Apr 26 14:31:45 2024
//+
Point(1) = {-0, 0, 0, 1.0};
//+
Point(2) = {-0, 0, 0, 1.0};
//+
Point(3) = {5, 0, 0, 1.0};
//+
Point(4) = {5, 1, 0, 1.0};
//+
Point(5) = {0, 1, 0, 1.0};
//+
Point(6) = {6, 0, 0, 1.0};
//+
Point(7) = {6, 1, 0, 1.0};
//+
Point(8) = {6, -3, 0, 1.0};
//+
Point(9) = {5, -3, 0, 1.0};
//+
Line(1) = {5, 1};
//+
Line(2) = {1, 3};
//+
Line(3) = {3, 9};
//+
Line(4) = {9, 8};
//+
Line(5) = {8, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 4};
//+
Line(8) = {4, 5};
//+
Curve Loop(1) = {8, 1, 2, 3, 4, 5, 6, 7};
//+
Plane Surface(1) = {1};
//+
Physical Curve("boundary", 9) = {8, 1, 2, 3, 7, 6, 5, 4};
Physical Surface("surface") = {1};
