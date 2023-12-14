import programl as pg
import cv2 as cv
import networkx as nx
import matplotlib.pyplot as plt
import pydot

G=pg.from_cpp("""
              #include <iostream>
              int main(int argc, char** argv){
                std::cout<<"Hello world!"<<std::endl;
                return 0;
              }
              """)

dot_string=pg.to_dot(G)
# print(dot_string)
graphs=pydot.graph_from_dot_data(dot_string)
graph=graphs[0]
graph.write_png("hello.png")