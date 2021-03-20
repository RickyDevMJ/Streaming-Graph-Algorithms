# Streaming-Graph-Algorithms
##How to run:
1. Clone the repository
2. src/common.mk contains system information such as library paths, compute capability etc. Update it for your system.
3. Run the makefile in src folder. This will create an executables in bin folder.
4. Run the executable as follows: ./scc_two_phase /path/to/base/graph /path/to/edge/updates
	1. The base graph can be in .mtx, .graph or .gr format.
	2. The edge update file must contain the list of edges to be added to the graph, with one edge in each line. 
	3. Both the files must contain the graph description in ***1-indexed*** form.
